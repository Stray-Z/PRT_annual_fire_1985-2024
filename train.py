import argparse
import logging
import os
from datetime import datetime

import torch
from prettytable import PrettyTable
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import CosineLRScheduler, create_scheduler, create_scheduler_v2
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
# from net.D_RMT import D_RMT_S, D_RMT_B
from net.HRNet import HighResolutionNet
# # from net.RMT_D import RMT_B_D, RMT_T_D, RMT_S_D
# from net.RMT import RMT_S
# from net.RMT_D import RMT_D_S
# from net.UHRNet_48w import UHRNet_48w
# from utils.RMT_scheduler import NoamLR,NoamOpt
from utils.lovasz_losses import lovasz_softmax
from utils import metric
from utils.loss import FocalLoss
# from net.uhrnet import UHRnet
from torch.utils.data import DataLoader
from dataset import GIDDataset
from UMNet import UMNet
# from allennlp.training.learning_rate_schedulers import NoamLR

import torch.nn.functional as F
import numpy as np


from tensorboardX import SummaryWriter


# checkpoint_filename = './ckpt/ARU-HRNet/2_test/ARU-HRNet_GID15_data.pth'
# checkpoint_filename = './ckpt/RMT/RMT_test/RMT_GID15_data.pth'
# checkpoint_filename = './ckpt/D-UHRNet/D-UHRNet_GID15_data.pth'
# checkpoint_filename = './ckpt/UHRNet/UHRNet_GID15_data.pth'
# checkpoint_filename = './ckpt/U-Net/U-Net_GID15_data.pth'
# checkpoint_filename = './ckpt/DeepLabV3+/DeepLabV3+_GID15_data.pth'
# checkpoint_filename = './ckpt/SegNet/SegNet_GID15_data.pth'
# checkpoint_filename = './ckpt/HRNet/HRNet_GID15_data.pth'

# mask为16个类别标签
# masks为8个类别标签

train_img_dir = r"/data/home2/train/image7"
train_mask_dir = r"/data/home2/train/mask7"
eval_img_dir = r"/data/home2/val/image7"
eval_mask_dir = r"/data/home2/val/mask7"

# train_img_dir = "mixdata/train/image"
# train_mask_dir = "mixdata/train/mask"
# eval_img_dir = "mixdata/val/image"
# eval_mask_dir = "mixdata/val/mask"
# tb_log = 'logs/ARU-HRNet'
# tb_log = 'logs/UHRNet'
tb_log = 'logs/U-Net'
# tb_log = 'logs/DeepLabV3+'
# tb_log = 'logs/SegNet'
# tb_log = 'logs/HRNet'
# tb_log = 'logs/DRMT'

# weight = [0.017718670650710446, 0.07212619929699658, 0.4015391063997188, 0.019372216880512445, 1.0, 0.07998996180176235,
          # 0.7757620124171001, 0.07555670795702805]


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num_classes', default=2, type=int, metavar='num_class')
    parser.add_argument('--batch-size', default=16, type=int)

    # parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
    #                     help='learning rate (default: 1e-3)')
    parser.add_argument('--device', default='cuda:0', type=str, metavar='DEVICE',
                        help='device to use for training / testing')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-9, metavar='W',
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 6e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-7, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--t_initial', type=float, default=40, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--cycle-decay', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Loss weight
    parser.add_argument('--ce_weight', default=1., type=float, metavar='CEWeight', help='weight of CrossEntropy')
    parser.add_argument('--focal_weight', default=0.5, type=float, metavar='FocalWeight', help='weight of FocalLoss')
    parser.add_argument('--lovasz_weight', default=0., type=float, metavar='LovaszWeight',
                        help='weight of LovaszSoftmaxLoss')
    parser.add_argument('--edge_weight', default=0., type=float, metavar='EdgeLossWeight', help='weight of EdgeLoss')
    parser.add_argument('--dice_weight', default=0., type=float, metavar='DiceLossWeight', help='weight of DiceLoss')
    # checkpoint_filename
    parser.add_argument('--ckp_path', type=str, default=r'/root/autodl-tmp/weight/Nets/weight/U-net_3.pth',
                        metavar='resume_from_checkpoint', help='checkpoint_filename')

    return parser


def configure_logging(log_file):
    # 设置logging的格式
    log_format = '%(asctime)s [%(levelname)s]: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,format=log_format,datefmt=date_format,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file)  # 添加一个FileHandler用于保存日志到文件中
                        ])
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def train(args):
    best_mIOU = 0.
    writer = SummaryWriter('logs/UNet')
    log_file = 'training_HRNet7.log'
    configure_logging(log_file)

    train_dataset = GIDDataset(train_img_dir, train_mask_dir, train=True)
    eval_dataset = GIDDataset(eval_img_dir, eval_mask_dir, train=False)

#新加入---------
    sample_img, sample_mask = train_dataset[0]
    print("Mask唯一值:", torch.unique(sample_mask))  # 应输出：tensor([-100, 0, 1])
    if -100 not in sample_mask:
        logging.warning("警告：mask 中未发现 -100，背景可能未被正确忽略！")
#--------------
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=8)

    # model = HighResolutionNet(args.num_classes)
    model = UMNet(3,2)
    # model = FasterNet(3,2)
    model = model.to(args.device)

    # criterion = nn.CrossEntropyLoss()
    # criterion1 = FocalLoss(gamma=2)
    # 损失函数
    # 修改损失函数（忽略背景-100）
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 保持原样
    criterion1 = FocalLoss(gamma=2,ignore_index=-100)  # 确保FocalLoss也忽略-100

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer=optimizer)

    conf_mat = np.zeros((args.num_classes, args.num_classes)).astype(np.int64)

    if args.ckp_path is not None and os.path.isfile(args.ckp_path):
        logging.info(f"Loading checkpoint '{args.ckp_path}'")
        checkpoint = torch.load(args.ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 1

    logging.info('Starting training')
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = 0
        model.train()
        with tqdm(total=len(train_dataloader.dataset), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for index, (img, mask) in enumerate(train_dataloader):
                img = img.to(args.device, dtype=torch.float32)
                mask = mask.to(args.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(img)
                output = F.softmax(outputs, dim=1)

                # Loss calculation
                lovasz_softmax_loss = lovasz_softmax(output, mask)
                focal_loss = criterion1(outputs, mask)
                loss = criterion(outputs, mask) + lovasz_softmax_loss + focal_loss

                loss.backward()
                optimizer.step()
                lr_scheduler.step(epoch)

                train_loss += loss.item()
                avg_train_loss = train_loss / len(train_dataloader)

                # values, preds = torch.max(output, 1)
                # preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                # masks = mask.data.cpu().numpy().squeeze().astype(np.uint8)
                # conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(), num_classes=args.num_classes)
                # 训练循环中的修改
                values, preds = torch.max(output, 1)
                preds = preds.cpu().numpy().squeeze().astype(np.uint8)  # 先移动到CPU再转NumPy
                masks = mask.cpu().numpy().squeeze().astype(np.uint8)  # 先移动到CPU再转NumPy

                # 仅计算有效像素（忽略-100）
                valid_pixels = (masks != -100)
                preds = preds[valid_pixels]
                masks = masks[valid_pixels]

                conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(),
                                                    num_classes=args.num_classes)

                pbar.update(img.shape[0])
                pbar.set_postfix({'loss (batch)': loss.item()})

        # Train metrics
        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa, train_mPA, train_mean_f1 ,avg_train_loss= metric.evaluate(conf_mat)

        writer.add_scalar('train_loss_per_epoch', avg_train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        writer.add_scalar('train_mIoU', train_mean_IoU, global_step=epoch)

        logging.info(f'train_mIoU: {train_mean_IoU}, train_Acc: {train_acc}, loss (batch): {avg_train_loss}')

        # Validation loop
        logging.info('Starting evaling')
        test_loss = 0
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(eval_dataloader.dataset), desc=f'Eval epoch {epoch}/{args.epochs}', unit='img') as pbar:
                for i, (img, mask) in enumerate(eval_dataloader):
                    img = img.to(args.device)
                    mask = mask.to(args.device)

                    outputs = model(img)
                    output = F.softmax(outputs, dim=1)

                    values, preds = torch.max(output, 1)
                    preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                    masks = mask.data.cpu().numpy().squeeze().astype(np.uint8)

                    # Loss calculation
                    lovasz_softmax_loss = lovasz_softmax(output, mask)
                    # focal_loss = criterion1(outputs, mask)
                    loss = criterion(outputs, mask) + lovasz_softmax_loss #+ focal_loss

                    test_loss += loss.item()

                    conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(), num_classes=args.num_classes)

                    pbar.update(img.shape[0])
                    pbar.set_postfix({'loss (batch)': loss.item()})

        # Validation metrics
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa, val_mPA, val_mean_f1 ,test_loss= metric.evaluate(conf_mat)

        writer.add_scalar('val_avg_loss', test_loss / len(eval_dataloader), epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_mIoU', val_mean_IoU, epoch)
        table = PrettyTable(["序号", "名称", "acc", "IoU"])

        # logging.info(f'Epoch {epoch} Val Per Class IOU \n' + PrettyTable(["序号", "名称", "acc", "IoU"]).get_string())

        for i in range(args.num_classes):
            table.add_row([i, train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i]])
        # print(table)
        # print("val_acc:", val_acc)
        # print("val_mean_IoU:", val_mean_IoU)
        # print("val_mPA: ", val_mPA)
        # print("val_mean_f1: ", val_mean_f1)
        # print("val_mean_recall: ", val_mean_recall)
        logging.info(f'Epoch {epoch} Val Per Class IOU \n' + table.get_string())
        # logging.info(f'Test Epoch {epoch}/{args.epochs} - Loss: {val_avg_loss}')


        if val_mean_IoU > best_mIOU:
            best_mIOU = val_mean_IoU
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_mIOU': best_mIOU
            }, args.ckp_path)
            writer.add_text('best_mIOU', f"best_mIOU: {best_mIOU}, best_mPA: {val_mPA}, best_mF1: {val_mean_f1}", global_step=epoch)
            logging.info(f'Saved checkpoint: {args.ckp_path}, at {epoch}')
            logging.info(f'Best_mIOU: {best_mIOU}, Best_Acc:{val_acc}, Best_mF1:{val_mean_f1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
