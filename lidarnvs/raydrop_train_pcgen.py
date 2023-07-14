import os
import numpy as np
import imageio
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
import pickle

l1loss = nn.L1Loss(reduction="mean")
mseloss = nn.MSELoss()
img2mse = lambda x, y: torch.mean((x - y) ** 2)
to8b = (
    lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    if np.max(x) < 10
    else (255.0 * np.clip(x / 81.0, 0, 1)).astype(np.uint8)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = -10 * np.log10(mse)  # max_value = 1
    return psnr


class RayDrop(nn.Module):
    def __init__(self, D=4, W=128, input_ch=3, output_ch=1):
        """ """
        super(RayDrop, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch

        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D - 1)]
        )
        self.output_linear = nn.Linear(W, output_ch)

        self.linears.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        return output


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--expname", type=str, default="raysdrop", help="experiment name"
    )
    parser.add_argument(
        "--basedir", type=str, default="./log", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data/usr/ziguo.tt/working/nerf/data/",
        help="input data directory",
    )

    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360", "nerfmvl"],
        help="The dataset loader to use.",
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")

    parser.add_argument(
        "--N_rand",
        type=int,
        default=2048,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=500,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options

    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=-1,
        help="set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical",
    )
    parser.add_argument(
        "--i_embed_views",
        type=int,
        default=-1,
        help="set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical",
    )

    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_save", type=int, default=1000, help="frequency of rays saving"
    )

    # lidar nerf
    parser.add_argument("--N_iters", type=int, default=500000)
    parser.add_argument("--H", type=int, default=66)
    parser.add_argument("--W", type=int, default=1030)

    # lr
    parser.add_argument("--cosLR", action="store_true")
    parser.add_argument(
        "--coslrate", type=float, default=5e-4, help="init learning rate"
    )
    parser.add_argument(
        "--cosminlrate", type=float, default=5e-5, help="min learning rate"
    )
    parser.add_argument("--warmup_iters", type=int, default=1000)

    # loss type

    parser.add_argument(
        "--rgb_loss_type",
        type=str,
        default="img2mse",
        help="options: img2mse / mseloss / l1loss",
    )

    return parser


def cosine_scheduler(
    base_value, final_value, globel_step, warmup_iters=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(globel_step - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == globel_step
    return schedule


def get_embedder(multires, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    elif i == 0:
        embed_kwargs = {
            "include_input": True,
            "input_dims": input_dims,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)
        out_dim = embedder_obj.out_dim
    return embed, out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            # embed_fns.append(lambda x : x/torch.norm(x, dim=-1, keepdim=True))
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def run_network(inputs, model, embed_fn, embeddirs_fn):
    """Prepares inputs and applies network 'fn'."""
    ray_direction, depth, intensity = inputs[:, :3], inputs[:, 3], inputs[:, 4]
    embedded_depth = embed_fn(depth.unsqueeze(1))
    embedded_intensity = embed_fn(intensity.unsqueeze(1))
    embedded_dirs = embeddirs_fn(ray_direction)
    input = torch.cat((embedded_dirs, embedded_depth, embedded_intensity), 1)
    outputs = model(input)
    return outputs


def load_pkl_data(data_dir, split):
    if not data_dir.is_dir():
        raise ValueError(f"Directory {data_dir} does not exist.")
    if split not in ["train", "test"]:
        raise ValueError(f"Split {split} not supported.")
    pkl_path = data_dir / f"{split}_data.pkl"
    if not pkl_path.is_file():
        raise ValueError(f"File {pkl_path} does not exist.")
    with open(pkl_path, "rb") as f:
        raydrop_data = pickle.load(f)
    return raydrop_data


def train():
    parser = config_parser()
    args = parser.parse_args()
    cosLR = args.cosLR
    loss_dict = {"img2mse": img2mse, "mseloss": mseloss, "l1loss": l1loss}

    H = args.H
    W = args.W

    # load dataset
    data_dir = Path(args.datadir)

    (directions, panos, intensities, raydrop_masks) = load_pkl_data(data_dir, "train")
    # print(
    #     np.array(directions).shape,
    #     np.array(panos).shape,
    #     np.array(intensities).shape,
    #     np.array(raydrop_masks).shape)
    rays_all = np.concatenate(
        (
            np.array(directions).reshape(-1, 3),
            np.array(panos).reshape(-1, 1),
            np.array(intensities).reshape(-1, 1),
        ),
        -1,
    )
    raydrop_masks = np.array(raydrop_masks)
    rays_all = rays_all[raydrop_masks.reshape(-1) > -1]
    raydrop_masks = np.where(raydrop_masks[raydrop_masks > -1] == 0.0, 0.0, 1.0)
    rays_all = np.concatenate((rays_all, raydrop_masks.reshape(-1, 1)), -1)

    (directions, panos, intensities, raydrop_masks) = load_pkl_data(data_dir, "test")
    raydrop_val_list = []
    for direction, pano, intensity, raydrop_mask in zip(
        directions, panos, intensities, raydrop_masks
    ):
        raydrop_val_list.append(
            np.concatenate(
                (
                    np.array(direction).reshape(-1, 3),
                    np.array(pano).reshape(-1, 1),
                    np.array(intensity).reshape(-1, 1),
                    np.array(raydrop_mask).reshape(-1, 1),
                ),
                -1,
            )
        )
    rays_val1 = raydrop_val_list[0]
    raydrop_masks = np.array(raydrop_masks)
    mask_val1 = np.where(raydrop_masks[0] > -1, 1, 0)
    ray_drop_val1 = raydrop_masks[0].reshape(H, W)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # network
    embed_fn, input_ch = get_embedder(args.multires, input_dims=1, i=args.i_embed)
    embeddirs_fn, input_ch_views = get_embedder(
        args.multires_views, input_dims=3, i=args.i_embed_views
    )
    total_input_ch = input_ch * 2 + input_ch_views
    # model
    model = RayDrop(D=args.netdepth, W=args.netwidth, input_ch=total_input_ch).to(
        device
    )
    grad_vars = list(model.parameters())
    # optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if f.endswith(".tar")
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])

    global_step = start

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        print("shuffle rays")
        np.random.shuffle(rays_all)
        rays_all = torch.Tensor(rays_all).to(device)
        rays_val1 = torch.Tensor(rays_val1).to(device)

    if args.render_test:
        print("RENDER ONLY")
        for idx, rays_val, raydrop_mask in zip(
            range(len(raydrop_val_list)), raydrop_val_list, raydrop_masks
        ):
            rays_val = torch.Tensor(rays_val).to(device)
            with torch.no_grad():
                predict_drop_val = run_network(rays_val, model, embed_fn, embeddirs_fn)
            imgbase = os.path.join(basedir, expname, str(idx))
            mask_bbox = np.where(raydrop_mask > -1, 1, 0)
            predict_drop_val = (
                np.where(predict_drop_val.cpu().numpy() > 0.5, 1.0, 0.0).reshape(H, W)
                * mask_bbox
            )
            np.save(imgbase + "_pred_drop.npy", predict_drop_val)
            imageio.imsave(imgbase + "_pred_drop.png", predict_drop_val.reshape(H, W))

            ray_drop_gt = np.where(raydrop_mask > 0, 1, 0)
            imageio.imsave(imgbase + "_gt_drop.png", ray_drop_gt.reshape(H, W))

        return

    N_iters = args.N_iters + 1
    print("Begin")

    loss_log = []
    val_psnr = []
    start = start + 1
    i_batch = 0

    lr_schedule = cosine_scheduler(
        base_value=args.coslrate,
        final_value=args.cosminlrate,
        globel_step=N_iters - 1,
        warmup_iters=args.warmup_iters,
    )

    for i in range(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_all[i_batch : i_batch + N_rand]  # [B, 2+1, 3*?]
            # ray_direction, depth, intensity, target_drop = batch[:, :3], batch[:, 3], batch[:, 4], batch[:, 5]
            inputs, target_drop = batch[:, :5], batch[:, 5]
            i_batch += N_rand
            if i_batch >= rays_all.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_all.shape[0])
                rays_all = rays_all[rand_idx]
                i_batch = 0

        #####  Core optimization loop  #####
        predict_drop = run_network(inputs, model, embed_fn, embeddirs_fn)
        optimizer.zero_grad()

        rgb_loss = loss_dict[args.rgb_loss_type]
        loss = rgb_loss(predict_drop, target_drop.unsqueeze(1))
        # loss = KL_loss_fun(predict_drop, target_drop.unsqueeze(1))

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ##   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            if cosLR:
                param_group["lr"] = lr_schedule[global_step]
            else:
                param_group["lr"] = new_lrate

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            ckpt = {
                "global_step": global_step,
                "network_fn_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, path)
            print("Saved checkpoints at", path)

        if i % args.i_save == 0 and i > 0:
            # Turn in testing mode
            with torch.no_grad():
                predict_drop_val = run_network(rays_val1, model, embed_fn, embeddirs_fn)
            imgbase = os.path.join(basedir, expname, "{:06d}_".format(i))
            predict_drop_val = (
                np.where(predict_drop_val.cpu().numpy() > 0.5, 1.0, 0.0).reshape(H, W)
                * mask_val1
            )
            psnr = cal_psnr(predict_drop_val.reshape(H, W), ray_drop_val1)
            print(psnr)
            val_psnr.append(psnr)
            loss_save = np.array(val_psnr)
            plt.plot(loss_save)
            plt.savefig(os.path.join(basedir, expname, "val_psnr.png"))
            plt.close()
            imageio.imsave(imgbase + "val_drop.png", predict_drop_val.reshape(H, W))

        loss_log.append(loss.item())
        if i % args.i_print == 0:
            loss_save = np.array(loss_log)
            plt.plot(loss_save)
            plt.savefig(os.path.join(basedir, expname, "loss_curve.png"))
            plt.close()

            loss_print = [loss.item()]

            print(f"[TRAIN] Iter: {i} Loss: {loss_print} ")

        global_step += 1
    loss_log = np.array(loss_log)
    np.save(os.path.join(basedir, expname, "loss_log.npy"), loss_log)
    val_psnr = np.array(val_psnr)
    np.save(os.path.join(basedir, expname, "val_psnr.npy"), val_psnr)


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    train()
