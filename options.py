# -*- coding: utf-8 -*-

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0", help="device")
parser.add_argument("--dataset", type=str, default="vessel", choices=["octa", "neural", "faros", "vessel2"], help="dataset")  # choices可扩展
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="train or test")

# data settings
parser.add_argument("--data_dir", type=str, default="D:/A_Project/OCTA-3M/data/faros/", help="path to folder for getting dataset")
parser.add_argument("--input_nc", type=int, default=3, choices=[1, 3], help="gray or rgb")
parser.add_argument("--crop_size", type=int, default=512, help="crop size")
parser.add_argument("--scale_size", type=int, default=512, help="scale size (applied in drive and cria)")
parser.add_argument("--output_ch", type=int, default=3, choices=[1, 3], help="output channel")

# training
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_workers", type=int, default=0, help="number of threads")
parser.add_argument("--val_epoch_freq", type=int, default=1, help="frequency of validation at the end of epochs")
parser.add_argument("--save_epoch_freq", type=int, default=5, help="frequency of saving models at the end of epochs")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--power", type=float, default=0.9, help="power")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
# first stage
parser.add_argument("--first_epochs", type=int, default=100, help="train epochs of first stage")

# results
parser.add_argument("--logs_dir", type=str, default="D:/A_Project/OCTA-3M/logs", help="path to folder for saving logs")
parser.add_argument("--models_dir", type=str, default="D:/A_Project/OCTA-3M/models", help="path to folder for saving models")
parser.add_argument("--results_dir", type=str, default="D:/A_Project/OCTA-3M/results", help="path to folder for saving results")
parser.add_argument("--suffix", type=str, default="best.pth", help="front_model-[model_suffix].pth will be loaded in models_dir")


args = parser.parse_args()
# args, unknown = parser.parse_known_args()
