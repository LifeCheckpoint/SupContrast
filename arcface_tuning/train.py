"""
@author: ronghuaiyang
@ref: https://github.com/ronghuaiyang/arcface-pytorch
"""

import os
import numpy as np
import time
import torch
import torchvision
import yaml
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model.focal_loss import *
from model.metrics import *
from model.resnet import *
from model.enc import *
from data.dataset import Dataset

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, "{}_{}.pth".format(name, iter_cnt))
    torch.save(model, save_name)
    return save_name

def list_images(directory):
    ls = []
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for i, subdir in enumerate(subdirs):
        subdir_path = os.path.join(directory, subdir)
        for file in os.listdir(subdir_path):
            if file.endswith(".png") or file.endswith(".jpg"):
                ls.append(str(os.path.join(subdir, file)) + " " + str(i + 1))
    return ls

def get_classes_num(directory):
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return len(subdirs) + 1

if __name__ == "__main__":
    # config
    try:
        with open("config.yml") as config_file:
            opt = yaml.load(config_file, Loader = yaml.FullLoader)
    except Exception as e:
        print(e)
        assert False, "Fail to read config file"

    # load config & prepare dataset
    device = torch.device("cuda")
    root = opt["train_root"]
    if not os.path.exists(opt["saving_path"]):
        os.makedirs(opt["saving_path"])

    train_dataset = Dataset(root, list_images(root))
    trainloader = data.DataLoader(train_dataset, batch_size = opt["train_batch_size"], shuffle = True, num_workers = opt["num_workers"])

    # loss
    if opt["loss"] == "focal_loss":
        criterion = FocalLoss(gamma = 2)
    elif opt["loss"] == "cross_entropy_loss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        assert False, "Invaild loss function: {}".format(opt["loss"])

    assert os.path.exists(opt["model_path"]), "{} does not exists.".format(opt["model_path"])
    model = FreezeEncoder(torch.load(opt["model_path"]))
    
    num_classes = get_classes_num(root)
    in_features = opt["model_features"]

    # metric
    if opt["metric"] == "add_margin":
        metric_fc = AddMarginProduct(in_features, num_classes, s = 30, m = 0.35)
    elif opt["metric"] == "arc_margin":
        metric_fc = ArcMarginProduct(in_features, num_classes, s = 30, m = 0.5)
    elif opt["metric"] == "sphere":
        metric_fc = SphereProduct(in_features, num_classes, m = 4)
    else:
        metric_fc = nn.Linear(in_features, num_classes)

    # optimizer & scheduler
    if opt["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([
            {"params": model.parameters()}, 
            {"params": metric_fc.parameters()}
        ], lr = opt["lr"], weight_decay = opt["weight_decay"])
    elif opt["optimizer"] == "Adam":
        optimizer = torch.optim.Adam([
            {"params": model.parameters()}, 
            {"params": metric_fc.parameters()}
        ], lr = opt["lr"], weight_decay = opt["weight_decay"])
    else:
        assert False, "illegal optimizer"
    scheduler = StepLR(optimizer, step_size = opt["lr_step"], gamma = 0.1)

    # start to train
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    print(str(model))
    print("train iters per epoch:".format(len(trainloader)))

    start = time.time()
    max_epoch = opt["max_epoch"]
    freeze_epoch = opt["freeze"]
    for i in range(max_epoch):
        scheduler.step()
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input, i + 1 >= freeze_epoch)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt["print_freq"] == 0:
                output = np.argmax(output.data.cpu().numpy(), axis = 1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt["print_freq"] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))

                print("{} train epoch {} iter {} {} iters/s loss {} acc {}".format(time_str, i, iters, speed, loss.item(), acc))

                start = time.time()

        if i % opt["save_interval"] == 0 or i == opt["max_epoch"]:
            save_model(model.get_backbone(), opt["saving_path"], opt["backbone"], i)
            print("model was saved.")
