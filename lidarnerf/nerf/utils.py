import glob
import os
import random
import time

import cv2
import imageio
import lpips
import mcubes
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import trimesh
from rich.console import Console
from skimage.metrics import structural_similarity
from torch_ema import ExponentialMovingAverage

from extern.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extern.fscore import fscore

from lidarnerf.dataset.base_dataset import custom_meshgrid

from lidarnerf.convert import pano_to_lidar


def is_ali_cluster():
    import socket

    hostname = socket.gethostname()
    return "auto-drive" in hostname


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def filter_bbox_dataset(pc, OBB_local):
    bbox_mask = np.isnan(pc[:, 0])
    z_min, z_max = min(OBB_local[:, 2]), max(OBB_local[:, 2])
    for i, (c1, c2) in enumerate(zip(pc[:, 2] <= z_max, pc[:, 2] >= z_min)):
        bbox_mask[i] = c1 and c2
    pc = pc[bbox_mask]
    OBB_local = sorted(OBB_local, key=lambda p: p[2])
    OBB_2D = np.array(OBB_local)[:4, :2]
    pc = filter_poly(pc, OBB_2D)
    return pc


def filter_poly(pcs, OBB_2D):
    OBB_2D = sort_quadrilateral(OBB_2D)
    mask = []
    for pc in pcs:
        mask.append(is_in_poly(pc[0], pc[1], OBB_2D))
    return pcs[mask]


def sort_quadrilateral(points):
    points = points.tolist()
    top_left = min(points, key=lambda p: p[0] + p[1])
    bottom_right = max(points, key=lambda p: p[0] + p[1])
    points.remove(top_left)
    points.remove(bottom_right)
    bottom_left, top_right = points
    if bottom_left[1] > top_right[1]:
        bottom_left, top_right = top_right, bottom_left
    return [top_left, top_right, bottom_right, bottom_left]


def is_in_poly(px, py, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f"[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}")

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (
            x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8
        )

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    )  # [S, 3]
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )  # [S, 1] --> [x, y, z]
                    u[
                        xi * S : xi * S + len(xs),
                        yi * S : yi * S + len(ys),
                        zi * S : zi * S + len(zs),
                    ] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = (
        vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
        + b_min_np[None, :]
    )
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.6f}"


class RMSEMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "RMSE"), self.measure(), global_step)

    def report(self):
        return f"RMSE = {self.measure():.6f}"


class MAEMeter:
    def __init__(self, intensity_inv_scale=1.0):
        self.V = 0
        self.N = 0
        self.intensity_inv_scale = intensity_inv_scale

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # Mean Absolute Error
        mae = np.abs(
            truths * self.intensity_inv_scale - preds * self.intensity_inv_scale
        ).mean()

        self.V += mae
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "MAE"), self.measure(), global_step)

    def report(self):
        return f"MAE = {self.measure():.6f}"


class DepthMeter:
    def __init__(self, scale):
        self.V = []
        self.N = 0
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error = self.compute_depth_errors(truths, preds)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1

    def compute_depth_errors(
        self, gt, pred, min_depth=1e-3, max_depth=80, thresh_set=1.25
    ):
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < thresh_set).mean()
        a2 = (thresh < thresh_set**2).mean()
        a3 = (thresh < thresh_set**3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )
        return rmse, a1, a2, a3, ssim

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(
            os.path.join(prefix, "depth error"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error(rmse, a1, a2, a3, ssim) = {self.measure()}"


class PointsMeter:
    def __init__(self, scale, intrinsics):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        chamLoss = chamfer_3DDist()
        pred_lidar = pano_to_lidar(preds[0], self.intrinsics)
        gt_lidar = pano_to_lidar(truths[0], self.intrinsics)

        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(pred_lidar[None, ...]).cuda(),
            torch.FloatTensor(gt_lidar[None, ...]).cuda(),
        )
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"CD f-score = {self.measure()}"


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def clear(self):
        self.V = 0
        self.N = 0

    # def prepare_inputs(self, *inputs):
    #     outputs = []
    #     for i, inp in enumerate(inputs):
    #         inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
    #         inp = inp.to(self.device)
    #         outputs.append(inp)
    #     return outputs

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        ssim = structural_similarity(
            preds.squeeze(0).squeeze(-1), truths.squeeze(0).squeeze(-1)
        )

        # preds, truths = self.prepare_inputs(
        #     preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        # ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f"SSIM = {self.measure():.6f}"


class LPIPSMeter:
    def __init__(self, net="alex", device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(
            truths, preds, normalize=True
        ).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(
            os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step
        )

    def report(self):
        return f"LPIPS ({self.net}) = {self.measure():.6f}"


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        opt,  # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        depth_metrics=[],
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=1,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.depth_metrics = depth_metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        # if self.opt.patch_size > 1:
        #     import lpips
        #     self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay
            )
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        # Initialize all returned values
        pred_intensity = None
        gt_intensity = None
        pred_depth = None
        gt_depth = None
        loss = 0

        if self.opt.enable_lidar:
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]

            images_lidar = data["images_lidar"]  # [B, N, 3/4]
            B_lidar, N_lidar, C_lidar = images_lidar.shape

            gt_raydrop = images_lidar[:, :, 0]
            gt_intensity = images_lidar[:, :, 1] * gt_raydrop
            gt_depth = images_lidar[:, :, 2] * gt_raydrop

            outputs_lidar = self.model.render(
                rays_o_lidar,
                rays_d_lidar,
                cal_lidar_color=True,
                staged=False,
                perturb=True,
                force_all_rays=False if self.opt.patch_size == 1 else True,
                **vars(self.opt),
            )

            pred_raydrop = outputs_lidar["image_lidar"][:, :, 0]
            pred_intensity = outputs_lidar["image_lidar"][:, :, 1] * gt_raydrop
            pred_depth = outputs_lidar["depth_lidar"] * gt_raydrop
            lidar_loss = (
                self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth)
                + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop)
                + self.opt.alpha_i
                * self.criterion["intensity"](pred_intensity, gt_intensity)
            )
            pred_intensity = pred_intensity.unsqueeze(-1)
            gt_intensity = gt_intensity.unsqueeze(-1)
        else:
            lidar_loss = 0

        loss = lidar_loss

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3:  # [K, B, N]
            loss = loss.mean(0)

        loss = loss.mean()

        if isinstance(self.opt.patch_size_lidar, int):
            patch_size_x, patch_size_y = (
                self.opt.patch_size_lidar,
                self.opt.patch_size_lidar,
            )
        elif len(self.opt.patch_size_lidar) == 1:
            patch_size_x, patch_size_y = (
                self.opt.patch_size_lidar[0],
                self.opt.patch_size_lidar[0],
            )
        else:
            patch_size_x, patch_size_y = self.opt.patch_size_lidar
        if self.opt.enable_lidar and patch_size_x > 1:
            pred_depth = (
                pred_depth.view(-1, patch_size_x, patch_size_y, 1)
                .permute(0, 3, 1, 2)
                .contiguous()
                / self.opt.scale
            )
            if self.opt.sobel_grad:
                pred_grad_x = F.conv2d(
                    pred_depth,
                    torch.tensor(
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device),
                    padding=1,
                )
                pred_grad_y = F.conv2d(
                    pred_depth,
                    torch.tensor(
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device),
                    padding=1,
                )
            else:
                pred_grad_y = torch.abs(
                    pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :]
                )
                pred_grad_x = torch.abs(
                    pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:]
                )

            dy = torch.abs(pred_grad_y)
            dx = torch.abs(pred_grad_x)

            if self.opt.grad_norm_smooth:
                grad_norm = torch.mean(torch.exp(-dx)) + torch.mean(torch.exp(-dy))
                # print('grad_norm', grad_norm)
                loss = loss + self.opt.alpha_grad_norm * grad_norm

            if self.opt.spatial_smooth:
                spatial_loss = torch.mean(dx**2) + torch.mean(dy**2)
                # print('spatial_loss', spatial_loss)
                loss = loss + self.opt.alpha_spatial * spatial_loss

            if self.opt.tv_loss:
                tv_loss = torch.mean(dx) + torch.mean(dy)
                # print('tv_loss', tv_loss)
                loss = loss + self.opt.alpha_tv * tv_loss

            if self.opt.grad_loss:
                gt_depth = (
                    gt_depth.view(-1, patch_size_x, patch_size_y, 1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    / self.opt.scale
                )
                gt_raydrop = (
                    gt_raydrop.view(-1, patch_size_x, patch_size_y, 1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

                # sobel
                if self.opt.sobel_grad:
                    gt_grad_y = F.conv2d(
                        gt_depth,
                        torch.tensor(
                            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.device),
                        padding=1,
                    )

                    gt_grad_x = F.conv2d(
                        gt_depth,
                        torch.tensor(
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.device),
                        padding=1,
                    )
                else:
                    gt_grad_y = gt_depth[:, :, :-1, :] - gt_depth[:, :, 1:, :]
                    gt_grad_x = gt_depth[:, :, :, :-1] - gt_depth[:, :, :, 1:]

                grad_clip_x = 0.01
                grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip_x, 1, 0)
                grad_clip_y = 0.01
                grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip_y, 1, 0)
                if self.opt.sobel_grad:
                    mask_dx = gt_raydrop * grad_mask_x
                    mask_dy = gt_raydrop * grad_mask_y
                else:
                    mask_dx = gt_raydrop[:, :, :, :-1] * grad_mask_x
                    mask_dy = gt_raydrop[:, :, :-1, :] * grad_mask_y

                if self.opt.depth_grad_loss == "cos":
                    patch_num = pred_grad_x.shape[0]
                    grad_loss = self.criterion["grad"](
                        (pred_grad_x * mask_dx).reshape(patch_num, -1),
                        (gt_grad_x * mask_dx).reshape(patch_num, -1),
                    )
                    grad_loss = 1 - grad_loss
                else:
                    grad_loss = self.criterion["grad"](
                        pred_grad_x * mask_dx, gt_grad_x * mask_dx
                    )
                loss = loss + self.opt.alpha_grad * grad_loss.mean()

        return (
            pred_intensity,
            gt_intensity,
            pred_depth,
            gt_depth,
            loss,
        )

    def eval_step(self, data):
        pred_intensity = None
        pred_depth = None
        pred_depth_crop = None
        pred_raydrop = None
        gt_intensity = None
        gt_depth = None
        gt_depth_crop = None
        gt_raydrop = None
        loss = 0
        if self.opt.enable_lidar:
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            images_lidar = data["images_lidar"]  # [B, H, W, 3/4]

            gt_raydrop = images_lidar[:, :, :, 0]
            if self.opt.dataloader == "nerf_mvl":
                valid_crop = gt_raydrop != -1
                valid_crop_idx = torch.nonzero(valid_crop)
                crop_h, crop_w = (
                    max(valid_crop_idx[:, 1]) - min(valid_crop_idx[:, 1]) + 1,
                    max(valid_crop_idx[:, 2]) - min(valid_crop_idx[:, 2]) + 1,
                )

                valid_mask = torch.where(gt_raydrop == -1, 0, 1)
                gt_raydrop = gt_raydrop * valid_mask

            gt_intensity = images_lidar[:, :, :, 1] * gt_raydrop
            gt_depth = images_lidar[:, :, :, 2] * gt_raydrop
            B_lidar, H_lidar, W_lidar, C_lidar = images_lidar.shape

            outputs_lidar = self.model.render(
                rays_o_lidar,
                rays_d_lidar,
                cal_lidar_color=True,
                staged=True,
                perturb=False,
                **vars(self.opt),
            )

            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(
                B_lidar, H_lidar, W_lidar, 2
            )
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]
            raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)
            if self.opt.dataloader == "nerf_mvl":
                raydrop_mask = raydrop_mask * valid_mask
            pred_intensity = pred_rgb_lidar[:, :, :, 1]
            pred_depth = outputs_lidar["depth_lidar"].reshape(B_lidar, H_lidar, W_lidar)
            # raydrop_mask = gt_raydrop  # TODO
            if self.opt.alpha_r > 0 and (not torch.all(raydrop_mask == 0)):
                pred_intensity = pred_intensity * raydrop_mask
                pred_depth = pred_depth * raydrop_mask

            lidar_loss = (
                self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth).mean()
                + self.opt.alpha_r
                * self.criterion["raydrop"](pred_raydrop, gt_raydrop).mean()
                + self.opt.alpha_i
                * self.criterion["intensity"](pred_intensity, gt_intensity).mean()
            )

            if self.opt.dataloader == "nerf_mvl":
                pred_intensity = pred_intensity[valid_crop].reshape(
                    B_lidar, crop_h, crop_w
                )
                gt_intensity = gt_intensity[valid_crop].reshape(B_lidar, crop_h, crop_w)
                pred_depth_crop = pred_depth[valid_crop].reshape(
                    B_lidar, crop_h, crop_w
                )
                gt_depth_crop = gt_depth[valid_crop].reshape(B_lidar, crop_h, crop_w)

            pred_intensity = pred_intensity.unsqueeze(-1)
            pred_raydrop = pred_raydrop.unsqueeze(-1)
            gt_intensity = gt_intensity.unsqueeze(-1)
            gt_raydrop = gt_raydrop.unsqueeze(-1)
        else:
            lidar_loss = 0

        loss = lidar_loss

        return (
            pred_intensity,
            pred_depth,
            pred_depth_crop,
            pred_raydrop,
            gt_intensity,
            gt_depth,
            gt_depth_crop,
            gt_raydrop,
            loss,
        )

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):
        pred_raydrop = None
        pred_intensity = None
        pred_depth = None

        if self.opt.enable_lidar:
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]
            outputs_lidar = self.model.render(
                rays_o_lidar,
                rays_d_lidar,
                cal_lidar_color=True,
                staged=True,
                perturb=perturb,
                **vars(self.opt),
            )

            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(
                -1, H_lidar, W_lidar, 2
            )
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]
            raydrop_mask = torch.where(pred_raydrop > 0.5, 1, 0)
            pred_intensity = pred_rgb_lidar[:, :, :, 1]
            pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)
            if self.opt.alpha_r > 0:
                pred_intensity = pred_intensity * raydrop_mask
                pred_depth = pred_depth * raydrop_mask

        return pred_raydrop, pred_intensity, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):
        if save_path is None:
            save_path = os.path.join(
                self.workspace, "meshes", f"{self.name}_{self.epoch}.ply"
            )

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))["sigma"]
            return sigma

        vertices, triangles = extract_geometry(
            self.model.aabb_infer[:3],
            self.model.aabb_infer[3:],
            resolution=resolution,
            threshold=threshold,
            query_func=query_func,
        )

        mesh = trimesh.Trimesh(
            vertices, triangles, process=False
        )  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            if is_ali_cluster() and self.opt.cluster_summary_path is not None:
                summary_path = self.opt.cluster_summary_path
            else:
                summary_path = os.path.join(self.workspace, "run", self.name)
            self.writer = tensorboardX.SummaryWriter(summary_path)

        change_dataloder = False
        if self.opt.change_patch_size_lidar[0] > 1:
            change_dataloder = True
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            if change_dataloder:
                if self.epoch % self.opt.change_patch_size_epoch == 0:
                    train_loader._data.patch_size_lidar = (
                        self.opt.change_patch_size_lidar
                    )
                    self.opt.patch_size_lidar = self.opt.change_patch_size_lidar
                else:
                    train_loader._data.patch_size_lidar = 1
                    self.opt.patch_size_lidar = 1

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds_raydrop, preds_intensity, preds_depth = self.test_step(data)

                if self.opt.enable_lidar:
                    pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                    pred_raydrop = (np.where(pred_raydrop > 0.5, 1.0, 0.0)).reshape(
                        loader._data.H_lidar, loader._data.W_lidar
                    )
                    pred_raydrop = (pred_raydrop * 255).astype(np.uint8)

                    pred_intensity = preds_intensity[0].detach().cpu().numpy()
                    pred_intensity = (pred_intensity * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_lidar = pano_to_lidar(
                        pred_depth / self.opt.scale, loader._data.intrinsics_lidar
                    )
                    if self.opt.dataloader == "nerf_mvl":
                        pred_lidar = filter_bbox_dataset(
                            pred_lidar, data["OBB_local"][:, :3]
                        )

                    np.save(
                        os.path.join(save_path, f"test_{name}_{i:04d}_depth_lidar.npy"),
                        pred_lidar,
                    )

                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    # pred_depth = (pred_depth / self.opt.scale).astype(np.uint8)

                    if write_video:
                        all_preds.append(cv2.applyColorMap(pred_intensity, 1))
                        all_preds_depth.append(cv2.applyColorMap(pred_depth, 9))
                    else:
                        cv2.imwrite(
                            os.path.join(save_path, f"test_{name}_{i:04d}_raydrop.png"),
                            pred_raydrop,
                        )
                        cv2.imwrite(
                            os.path.join(
                                save_path, f"test_{name}_{i:04d}_intensity.png"
                            ),
                            cv2.applyColorMap(pred_intensity, 1),
                        )
                        cv2.imwrite(
                            os.path.join(save_path, f"test_{name}_{i:04d}_depth.png"),
                            cv2.applyColorMap(pred_depth, 9),
                        )

                pbar.update(loader.batch_size)

        if write_video:
            if self.opt.enable_lidar:
                all_preds = np.stack(all_preds, axis=0)
                all_preds_depth = np.stack(all_preds_depth, axis=0)
                imageio.mimwrite(
                    os.path.join(save_path, f"{name}_lidar_rgb.mp4"),
                    all_preds,
                    fps=25,
                    quality=8,
                    macro_block_size=1,
                )
                imageio.mimwrite(
                    os.path.join(save_path, f"{name}_depth.mp4"),
                    all_preds_depth,
                    fps=25,
                    quality=8,
                    macro_block_size=1,
                )

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
            for metric in self.depth_metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                (
                    pred_intensity,
                    gt_intensity,
                    pred_depth,
                    gt_depth,
                    loss,
                ) = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for i, metric in enumerate(self.depth_metrics):
                        if i < 2:  # hard code
                            metric.update(pred_intensity, gt_intensity)
                        else:
                            metric.update(pred_depth, gt_depth)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.depth_metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="LiDAR_train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()
            for metric in self.depth_metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    (
                        preds_intensity,
                        preds_depth,
                        preds_depth_crop,
                        preds_raydrop,
                        gt_intensity,
                        gt_depth,
                        gt_depth_crop,
                        gt_raydrop,
                        loss,
                    ) = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [
                        torch.zeros_like(preds).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [
                        torch.zeros_like(preds_depth).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [
                        torch.zeros_like(truths).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    for i, metric in enumerate(self.depth_metrics):
                        if i < 2:  # hard code
                            metric.update(preds_intensity, gt_intensity)
                        else:
                            if (
                                self.opt.dataloader == "nerf_mvl" and i == 2
                            ):  # hard code
                                metric.update(preds_depth_crop, gt_depth_crop)
                            else:
                                metric.update(preds_depth, gt_depth)

                    if self.opt.enable_lidar:
                        save_path_raydrop = os.path.join(
                            self.workspace,
                            "validation",
                            f"{name}_{self.local_step:04d}_rarydrop.png",
                        )
                        save_path_intensity = os.path.join(
                            self.workspace,
                            "validation",
                            f"{name}_{self.local_step:04d}_intensity.png",
                        )
                        save_path_depth = os.path.join(
                            self.workspace,
                            "validation",
                            f"{name}_{self.local_step:04d}_depth.png",
                        )
                        os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)

                        pred_intensity = preds_intensity[0].detach().cpu().numpy()
                        pred_intensity = (pred_intensity * 255).astype(np.uint8)

                        pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                        pred_raydrop = (np.where(pred_raydrop > 0.5, 1.0, 0.0)).reshape(
                            loader._data.H_lidar, loader._data.W_lidar
                        )
                        pred_raydrop = (pred_raydrop * 255).astype(np.uint8)

                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        pred_lidar = pano_to_lidar(
                            pred_depth / self.opt.scale, loader._data.intrinsics_lidar
                        )
                        pred_depth = (pred_depth * 255).astype(np.uint8)
                        # pred_depth = (pred_depth / self.opt.scale).astype(np.uint8)

                        # cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_raydrop, pred_raydrop)
                        cv2.imwrite(
                            save_path_intensity, cv2.applyColorMap(pred_intensity, 1)
                        )
                        cv2.imwrite(save_path_depth, cv2.applyColorMap(pred_depth, 9))
                        np.save(
                            os.path.join(
                                self.workspace,
                                "validation",
                                f"{name}_{self.local_step:04d}_lidar.npy",
                            ),
                            pred_lidar,
                        )

                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if len(self.depth_metrics) > 0:
                # result = self.metrics[0].measure()
                result = self.depth_metrics[-1].measure()[0]  # hard code
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            for metric in self.depth_metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="LiDAR_evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:
            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if "density_grid" in state["model"]:
                        del state["model"]["density_grid"]

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if model_only:
            return

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
