# -*- coding: utf-8 -*-
import skimage.io as io
import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import morphology

from torchvision import transforms
from torchvision.utils import save_image

import helper
from evaluation import *


def get_results(loss_lst, auc_lst, acc_lst, sen_lst, fdr_lst, spe_lst, bacc_lst, kappa_lst, gmean_lst, iou_lst, dice_lst,
                criterion, pred, mask):

    pred[pred < 0.45] = 0
    pred[pred > 0.45] = 1
    # loss_lst.append((criterion1(pred, mask) + criterion2(pred, mask)).item())
    loss_lst.append(criterion(pred, mask).item())

    pred_arr = pred.squeeze().cpu().detach().numpy()
    mask_array = mask.squeeze().cpu().numpy()
    # mask_array = helper.mask2onehot(np.array(mask.squeeze().cpu()), num_classes=3)

    auc_lst.append(calc_auc(pred_arr, mask_array))

    '''pred_arr = np.array(pred_arr, bool)
    pred_arr = morphology.remove_small_objects(pred_arr, min_size=16, connectivity=2, in_place=True)
    pred_arr = np.array(pred_arr, np.uint8)'''

    acc_lst.append(calc_acc(pred_arr, mask_array))
    sen_lst.append(calc_sen(pred_arr, mask_array))
    fdr_lst.append(calc_fdr(pred_arr, mask_array))
    spe_lst.append(calc_spe(pred_arr, mask_array))
    bacc_lst.append(calc_bacc(pred_arr, mask_array))
    kappa_lst.append(calc_kappa(pred_arr, mask_array))
    gmean_lst.append(calc_gmean(pred_arr, mask_array))
    iou_lst.append(calc_iou(pred_arr, mask_array))
    dice_lst.append(calc_dice(pred_arr, mask_array))


    '''# OTSU threshold
    loss_lst.append(criterion(pred, mask).item())

    pred_arr = pred.squeeze().cpu().numpy()
    mask_array = mask.squeeze().cpu().numpy()
    auc_lst.append(calc_auc(pred_arr, mask_array))

    pred_img = np.array(pred_arr * 255, np.uint8)
    mask_img = np.array(mask_array * 255, np.uint8)

    # thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh_pred_img = cv2.adaptiveThreshold(pred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -50)
    thresh_pred_img = pred_img
    thresh_pred_img[thresh_pred_img >= 100] = 255
    thresh_pred_img[thresh_pred_img < 100] = 0
    thresh_pred_img = np.array(thresh_pred_img / 255, bool)
    thresh_pred_img = morphology.remove_small_objects(thresh_pred_img, min_size=16, connectivity=2, in_place=True)
    thresh_pred_img = np.array(thresh_pred_img * 255, np.uint8)

    acc_lst.append(calc_acc(thresh_pred_img / 255.0, mask_img / 255.0))
    sen_lst.append(calc_sen(thresh_pred_img / 255.0, mask_img / 255.0))
    fdr_lst.append(calc_fdr(thresh_pred_img / 255.0, mask_img / 255.0))
    spe_lst.append(calc_spe(thresh_pred_img / 255.0, mask_img / 255.0))
    bacc_lst.append(calc_bacc(thresh_pred_img / 255.0, mask_img / 255.0))
    kappa_lst.append(calc_kappa(thresh_pred_img / 255.0, mask_img / 255.0))
    gmean_lst.append(calc_gmean(thresh_pred_img / 255.0, mask_img / 255.0))
    iou_lst.append(calc_iou(thresh_pred_img / 255.0, mask_img / 255.0))
    dice_lst.append(calc_dice(thresh_pred_img / 255.0, mask_img / 255.0))'''


def print_results(loss_lst, auc_lst, acc_lst, sen_lst, fdr_lst, spe_lst, bacc_lst, kappa_lst, gmean_lst, iou_lst, dice_lst):
    loss_arr = np.array(loss_lst)
    auc_arr = np.array(auc_lst)
    acc_arr = np.array(acc_lst)
    sen_arr = np.array(sen_lst)
    fdr_arr = np.array(fdr_lst)
    spe_arr = np.array(spe_lst)
    bacc_arr = np.array(bacc_lst)
    kappa_arr = np.array(kappa_lst)
    gmean_arr = np.array(gmean_lst)
    iou_arr = np.array(iou_lst)
    dice_arr = np.array(dice_lst)

    print("Loss - mean: " + str(loss_arr.mean()) + "\tstd: " + str(loss_arr.std()))
    print("AUC - mean: " + str(auc_arr.mean()) + "\tstd: " + str(auc_arr.std()))
    print("ACC - mean: " + str(acc_arr.mean()) + "\tstd: " + str(acc_arr.std()))
    print("SEN - mean: " + str(sen_arr.mean()) + "\tstd: " + str(sen_arr.std()))
    print("FDR - mean: " + str(fdr_arr.mean()) + "\tstd: " + str(fdr_arr.std()))
    print("SPE - mean: " + str(spe_arr.mean()) + "\tstd: " + str(spe_arr.std()))
    print("BACC - mean: " + str(bacc_arr.mean()) + "\tstd: " + str(bacc_arr.std()))
    print("Kappa - mean: " + str(kappa_arr.mean()) + "\tstd: " + str(kappa_arr.std()))
    print("G-mean - mean: " + str(gmean_arr.mean()) + "\tstd: " + str(gmean_arr.std()))
    print("IOU - mean: " + str(iou_arr.mean()) + "\tstd: " + str(iou_arr.std()))
    print("Dice - mean: " + str(dice_arr.mean()) + "\tstd: " + str(dice_arr.std()))

    return loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, bacc_arr, kappa_arr, gmean_arr, iou_arr, dice_arr


def test(dataloader, net, device, criterion):
    loss_dct = {"octa": []}  # Loss
    auc_dct = {"octa": []}  # AUC
    acc_dct = {"octa": []}  # Accuracy
    sen_dct = {"octa": []}  # Sensitivity (Recall)
    fdr_dct = {"octa": []}  # FDR
    spe_dct = {"octa": []}  # Specificity
    bacc_dct = {"octa": []}  # Balanced Accuracy
    kappa_dct = {"octa": []}  # Kappa
    gmean_dct = {"octa": []}  # G-mean
    iou_dct = {"octa": []}  # IOU
    dice_dct = {"octa": []}  # Dice Coefficient (F1-score)

    i = 1
    with torch.no_grad():
        for sample in dataloader:
            if len(sample) != 2:
                print("Error occured in sample %03d, skip" % i)
                continue

            print("Evaluate %03d..." % i)
            i += 1

            img = sample[0].to(device)
            mask = sample[1].to(device)
            # with torch.cuda.amp.autocast():
            pred = net(img)

            new = '%03d' % (i - 1)
            name1 = 'D:/A_Project/OCTA-3M/results/octa/img/' + new + ".png"
            name2 = 'D:/A_Project/OCTA-3M/results/octa/mask/' + new + ".png"
            name3 = 'D:/A_Project/OCTA-3M/results/octa/pred/' + new + ".png"
            save_image(img[0, :, :, :], name1)
            save_image(mask[0, :, :, :], name2)
            save_image(pred[0, :, :, :], name3)

            get_results(loss_dct["octa"], auc_dct["octa"], acc_dct["octa"], sen_dct["octa"],
                        fdr_dct["octa"], spe_dct["octa"], bacc_dct["octa"], kappa_dct["octa"], gmean_dct["octa"],
                        iou_dct["octa"], dice_dct["octa"], criterion, pred, mask)

    loss_dct["octa"], auc_dct["octa"], acc_dct["octa"], sen_dct["octa"], \
    fdr_dct["octa"], spe_dct["octa"], bacc_dct["octa"], kappa_dct["octa"], gmean_dct["octa"], \
    iou_dct["octa"], dice_dct["octa"] = print_results(loss_dct["octa"], auc_dct["octa"], acc_dct["octa"],
                                                      sen_dct["octa"], fdr_dct["octa"], spe_dct["octa"],
                                                      bacc_dct["octa"], kappa_dct["octa"], gmean_dct["octa"],
                                                      iou_dct["octa"], dice_dct["octa"])

    return loss_dct, auc_dct, acc_dct, sen_dct, fdr_dct, spe_dct, bacc_dct, kappa_dct, gmean_dct, iou_dct, dice_dct
