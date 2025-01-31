import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
import copy
from datetime import datetime
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder, Vimeo90kDataset

# from src.models.image_model import IntraNoAR
from src.models.waseda_vr import Cheng2020Anchor
from src.zoo.image import model_architectures as architectures
from pytorch_msssim import ms_ssim as MS_SSIM


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbdas=[435.6675, 845.325, 1625.625, 3140.7075]):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.mse2=nn.MSELoss(reduction='none')
        self.lmbdas = lmbdas

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}
        out["bpp_loss"] = sum((torch.log(likelihoods).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels))
                              for likelihoods in output["likelihoods"].values())
        out["mse_loss"] = self.mse2(output["x_hat"], target).mean(dim=[1, 2, 3])
        # out["msssim_loss"] = 1 - MS_SSIM(output["x_hat"], target, data_range=1.0, size_average=False)
        # print(output["lmb"].shape, out["mse_loss"].shape, out["bpp_loss"].shape)
        out["loss"] = torch.mean(output["lmb"] * out["mse_loss"] + out["bpp_loss"])

        return out


class MaxLengthList():
    def __init__(self, max_len, dtype=np.float32):
        self._list = np.empty(0, dtype=dtype)
        self._max_len = int(max_len)
        self._next_idx = int(0)

    def add(self, v):
        v = float(v)
        _len = len(self._list)
        if _len < self._max_len:
            self._list = np.append(self._list, v)
        else:
            assert _len == self._max_len, f'invalid length={_len}, max_len={self._max_len}'
            self._list[self._next_idx] = v
            self._next_idx = (self._next_idx + 1) % self._max_len

    def current(self) -> float:
        if len(self._list) == 0:
            print(f'Warning: the length of self._list={self._list} is 0')
            return None
        return self._list[self._next_idx - 1]

    def median(self) -> float:
        return np.median(self._list)

    def max(self) -> float:
        return np.max(self._list)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    emb_params = []
    net_params = []
    aux_params = []
    for n, p in net.named_parameters():
        if n.endswith(".quantiles"):
            p.requires_grad = True
            aux_params.append(p)
        elif "embedding" in n:
            p.requires_grad = True
            net_params.append(p)
        # elif n.startswith(tuple(["g_a", "g_s", "h_a", "h_s", "entropy_parameters"])):
        else:
            p.requires_grad = True
            net_params.append(p)
    optimizer = optim.Adam(
        net_params,
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        aux_params, lr=args.aux_learning_rate
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    args, model, criterion, train_dataloader, optimizer, aux_optimizer, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device
    _moving_grad_norm_buffer = MaxLengthList(max_len=100)

    for i, d in enumerate(train_dataloader):

        d = d.to(device)
        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()
        
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.learning_rate

        aux_loss = model.aux_loss()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.aux_parameters(), clip_max_norm)
        aux_optimizer.step()
        
        if i*len(d) % 5000 == 0:
            print("===grad===")
            for name, parms in model.named_parameters():
                print('-->name:', name)
                # print('-->para:', parms)
                print('-->grad_requirs:', parms.requires_grad)
                if parms.grad != None:
                    print('-->grad_value:', parms.grad.mean(), parms.grad.min(), parms.grad.max())
                    print('-->grad_fn:', parms.grad_fn)
                else:
                    print('-->grad_value: None')
                print("===")
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        if i*len(d) % 5000 == 0:
            logging.info(
                f'[{i*len(d)}/{len(train_dataloader.dataset)}] | '
                # f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'Loss: {out_criterion["loss"].mean().item():.3f} | '
                f'MSE loss: {out_criterion["mse_loss"].mean().item():.5f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].mean().item():.4f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )

def update_modules(model, force=True):
    for name, module in model.named_modules():
        if hasattr(module, 'update') and callable(getattr(module, 'update')):
            print(f"Module '{name}' has an update method.")
            try:
                module.update(force=force)
            except TypeError:
                module.update()

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    # update_modules(model, force=True)
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].mean())
            loss.update(out_criterion["loss"].mean())
            mse_loss.update(out_criterion["mse_loss"].mean())

    logging.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg.item():.5f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="cheng2020_vr",
        type=str,
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality-level",
        type=int,
        default=11,
        help="Quality level (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-aux_lr",
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    # train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataset = Vimeo90kDataset(args.dataset, split="train", transform=train_transforms, tuplet=3)
    test_dataset = Vimeo90kDataset(args.dataset, split="valid", transform=test_transforms, tuplet=3)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = Cheng2020Anchor(N=192)
    nk = []
    for n, k in net.named_parameters():
        nk.append(n)
    # net = architectures["cheng2020-anchor"]
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        net = net.from_state_dict(checkpoint["state_dict"])
        update_modules(net)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,], gamma=0.1)
    criterion = RateDistortionLoss()

    last_epoch = 0

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======'%epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            args,
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer, 
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(), 
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir
            )


if __name__ == "__main__":
    main(sys.argv[1:])
