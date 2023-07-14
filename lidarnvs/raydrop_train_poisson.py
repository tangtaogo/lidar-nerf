import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from lidarnvs.raydrop_dataset_poisson import RaydropDataset
from lidarnvs.unet import UNet, dice_coeff, dice_loss, multiclass_dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # Iterate over the test set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Test round",
            unit="batch",
            leave=False,
        ):
            images, true_masks = batch

            # Move images and labels to correct device and type
            images = images.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(images)
            true_masks = true_masks.reshape(mask_pred.shape)

            if net.n_classes == 1:
                assert (
                    true_masks.min() >= 0 and true_masks.max() <= 1
                ), "True mask indices should be in [0, 1]"
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # Compute the dice score
                dice_score += dice_coeff(
                    mask_pred, true_masks, reduce_batch_first=False
                )
            else:
                assert (
                    true_masks.min() >= 0 and true_masks.max() < net.n_classes
                ), "True mask indices should be in [0, n_classes["
                # Convert to one-hot format
                true_masks = (
                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                )
                mask_pred = (
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                # Compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:], true_masks[:, 1:], reduce_batch_first=False
                )

    net.train()
    return dice_score / max(num_val_batches, 1)


def train_model(
    model,
    data_dir,
    ckpt_dir,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    data_dir = Path(data_dir)
    ckpt_dir = Path(ckpt_dir)

    # Create dataset
    train_dataset = RaydropDataset(data_dir=data_dir, split="train")
    test_dataset = RaydropDataset(data_dir=data_dir, split="test")
    n_train = len(train_dataset)
    n_test = len(test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=RaydropDataset.collate_fn,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=RaydropDataset.collate_fn,
        shuffle=True,
    )

    # Initialize logging
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    experiment.config.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "save_checkpoint": save_checkpoint,
            "img_scale": img_scale,
            "amp": amp,
        }
    )

    log_str = (
        f"Starting training:\n"
        f"Epochs:          {epochs}\n"
        f"Batch size:      {batch_size}\n"
        f"Learning rate:   {learning_rate}\n"
        f"Training size:   {n_train}\n"
        f"Validation size: {n_test}\n"
        f"Checkpoints:     {save_checkpoint}\n"
        f"Device:          {device.type}\n"
        f"Images scaling:  {img_scale}\n"
        f"Mixed Precision: {amp}\n"
    )
    logging.info(log_str)

    # Set up optimizer, loss, lr_scheduler, loss scaling.
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch

                if images.shape[1] != model.n_channels:
                    raise ValueError(
                        f"Input channel mismatch: "
                        f"{images.shape[1]} vs {model.n_channels}"
                    )
                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(
                    device.type if device.type != "mps" else "cpu", enabled=amp
                ):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(
                            F.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                        )
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes)
                            .permute(0, 3, 1, 2)
                            .float(),
                            multiclass=True,
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                )
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                division_step = n_train // (5 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace("/", ".")
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms["Weights/" + tag] = wandb.Histogram(
                                    value.data.cpu()
                                )
                            if not (
                                torch.isinf(value.grad) | torch.isnan(value.grad)
                            ).any():
                                histograms["Gradients/" + tag] = wandb.Histogram(
                                    value.grad.data.cpu()
                                )

                        val_score = evaluate(model, test_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info("Validation Dice score: {}".format(val_score))
                        try:
                            experiment.log(
                                {
                                    "learning rate": optimizer.param_groups[0]["lr"],
                                    "validation Dice": val_score,
                                    "images": wandb.Image(images[0].cpu()),
                                    "masks": {
                                        "true": wandb.Image(
                                            true_masks[0].float().cpu()
                                        ),
                                        "pred": wandb.Image(
                                            masks_pred.argmax(dim=1)[0].float().cpu()
                                        ),
                                    },
                                    "step": global_step,
                                    "epoch": epoch,
                                    **histograms,
                                }
                            )
                        except:
                            pass

        if save_checkpoint:
            checkpoint_path = ckpt_dir / f"checkpoint_{epoch:03}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoint_path)
            logging.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--data_dir", type=str, default="N/A", help="Path to the raydrop dataset."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="N/A", help="Path to the checkpoint directory."
    )
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--batch-size", "-b", dest="batch_size", type=int, default=2, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=1, help="Number of classes"
    )

    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # - n_channels: 10
    #   - [0]   : hit_masks
    #   - [1]   : hit_depths
    #   - [2:5] : hit_normals (world), TODO: change to local coord
    #   - [5]   : hit_incidences in cosine
    #   - [6]   : intensities
    #   - [7:10]: rays_d, TODO: change to local coord
    # - n_classes: 1
    #   - number of probabilities you want to get per pixel
    #   - raydrop_masks has only 1 channel
    model = UNet(n_channels=10, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    model.to(device=device)
    train_model(
        model=model,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
