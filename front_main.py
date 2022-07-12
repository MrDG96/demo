# -*- coding: utf-8 -*-

import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from iouLoss import IOU
from losses import SoftDiceLossV2, SoftDiceLoss, focal_loss, LovaszLossSoftmax, AsymmetricLossOptimized, AsymmetricLoss, \
    ASLSingleLabel
from msssimLoss import MSSSIM
from options import args
from utils import mkdir, build_dataset, Visualizer  # build_model,
from first_stage import ResNest_UNet3, ResNest_UNet, UNet_3Plus, UNet

from train import train
from val import val
from test import test

# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == "train":
    isTraining = True
    isTesting = False
else:
    isTraining = False
    isTesting = True

database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining, isTesting=isTesting,
                         crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))

if isTraining:  # train
    NAME = args.dataset  # + args.model + "_" + args.loss
    viz = Visualizer(env=NAME)
    writer = SummaryWriter(args.logs_dir)
    mkdir(args.models_dir)

    # 加载数据集
    train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=False,
                                 isTesting=False, crop_size=(args.crop_size, args.crop_size),
                                 scale_size=(args.scale_size, args.scale_size))
    val_dataloader = DataLoader(val_database, batch_size=1)

    # 构建模型
    first_net = ResNest_UNet3(img_ch=args.input_nc, output_ch=args.output_ch).to(device)
    first_net = torch.nn.DataParallel(first_net)
    first_optim = optim.Adam(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

    # criterion = AsymmetricLossOptimized()
    # criterion = torch.nn.BCELoss()
    # criterion = focal_loss()
    # criterion = MSSSIM()
    # criterion3 = IOU()
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = SoftDiceLoss()  # 可更改

    best_octa = {"epoch": 0, "dice": 0, "auc": 0}

    # start training
    print("Start training...")
    for epoch in range(args.first_epochs):
        print('Epoch %d / %d' % (epoch + 1, args.first_epochs))
        print('-' * 10)
        first_net = train(viz, writer, train_dataloader, first_net, first_optim, args.init_lr, criterion, device,
                          args.power, epoch, args.first_epochs)
        if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.first_epochs - 1:
            first_net, best_octa = val(best_octa, viz, writer, val_dataloader, first_net,
                                       criterion, device, args.save_epoch_freq,
                                       args.models_dir, epoch, args.first_epochs)
    print("Training finished.")
else:  # test
    # 加载数据集和模型
    test_dataloader = DataLoader(database, batch_size=1)
    net = torch.load(args.models_dir + "/model-" + args.suffix).to(device)
    net.eval()
    # criterion = torch.nn.BCELoss()
    # criterion = focal_loss()
    criterion = torch.nn.MSELoss()  # 可更改
    # start testing
    print("Start testing...")
    test(test_dataloader, net, device, criterion)
    print("Testing finished.")
