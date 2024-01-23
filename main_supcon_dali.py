from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import requests
from torchvision import transforms, datasets
import torch.nn as nn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

PUSH_MEG = False
MSG_TOKEN = ""
COMPILE_MODEL = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    assert opt.data_folder is not None \
        and opt.mean is not None \
        and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/path_models'
    opt.tb_path = './save/SupCon/path_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, "path", opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


class CustomPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_folder, mean, std, size):
        super(CustomPipeline, self).__init__(batch_size, num_threads, device_id, seed=42)

        # 使用 DALI 中的图像加载和增强操作
        self.input = ops.FileReader(file_root=data_folder, shard_id=device_id, num_shards=1, random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device='gpu', size=size)
        self.flip = ops.Flip(device='gpu')
        self.brightness_contrast = ops.BrightnessContrast(device='gpu', brightness=0.4, contrast=0.4)
        self.transpose = ops.Transpose(device='gpu')
        self.to_tensor = ops.ToTensor(device='gpu')
        self.normalize = ops.NormalizePermute(device='gpu', mean=mean, std=std)

    def define_graph(self):
        # 定义 DALI 图像处理流程
        images, labels = self.input()
        images = self.decode(images)
        images = self.res(images)
        images = self.flip(images)
        images = self.brightness_contrast(images)
        images = self.transpose(images)
        images = self.to_tensor(images)
        images = self.normalize(images)

        return images, labels

def set_loader(opt):
    # 使用 DALI 数据加载和增强
    train_loader = CustomPipeline(
        batch_size=opt.batch_size, num_threads=opt.num_workers, device_id=0,
        data_folder=opt.data_folder, mean=eval(opt.mean), std=eval(opt.std), size=opt.size)

    # 构建 DALI DataLoader
    train_loader.build()
    dali_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, num_workers=opt.num_workers)

    return dali_loader
    
"""
def set_loader(opt):
    normalize = transforms.Normalize(mean=eval(opt.mean), std=eval(opt.std))

    # 定义 ToTensor 操作并将其移到 GPU 上
    to_tensor = transforms.ToTensor()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.3, 0.1, 0.1)
        ], p=0.8),
        to_tensor,  # 使用在 GPU 上定义的 to_tensor 操作
        normalize,
    ])

    train_dataset = datasets.ImageFolder(root=opt.data_folder, transform=TwoCropTransform(train_transform))

    # 使用 pin_memory 加速数据传输到 GPU
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    return train_loader
"""


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        if COMPILE_MODEL:
            model = torch.compile(model, mode = "reduce-overhead")
            criterion = torch.compile(criterion, mode = "reduce-overhead")
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 将图像数据移到 GPU 上
        images = torch.cat([images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)], dim=0)
        
        # 以下保持不变
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        loss = criterion(features)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # 使用 DALI 数据加载和增强
    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
            if PUSH_MEG:
                requests.get("http://www.pushplus.plus/send?token={}&title=TrainingProcess&content=**{} epochs**, **loss = {}**&template=markdown".format(MSG_TOKEN, epoch, str(loss)))

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()    

"""
python main_supcon_dali.py --batch_size 2048  --learning_rate 0.5  --temp 0.1 --cosine --data_folder EmojiDataset --mean "(0.5, 0.5, 0.5)" --std "(0.25, 0.25, 0.25)" --method SimCLR --save_freq 25
"""
