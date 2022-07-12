# -*- coding: utf-8 -*-

import os
import torch

from utils import mkdir
from test import test


def visual_results(loss_arr, auc_arr, acc_arr, sen_arr, fdr_arr, spe_arr, bacc_arr, kappa_arr, gmean_arr, iou_arr, dice_arr, viz,
                   writer, epoch, flag=""):
    loss_mean = loss_arr.mean()
    auc_mean = auc_arr.mean()
    acc_mean = acc_arr.mean()
    sen_mean = sen_arr.mean()
    fdr_mean = fdr_arr.mean()
    spe_mean = spe_arr.mean()
    bacc_mean = bacc_arr.mean()
    kappa_mean = kappa_arr.mean()
    gmean_mean = gmean_arr.mean()
    iou_mean = iou_arr.mean()
    dice_mean = dice_arr.mean()

    viz.plot("val" + flag + " loss", loss_mean)
    viz.plot("val" + flag + " auc", auc_mean)
    viz.plot("val" + flag + " acc", acc_mean)
    viz.plot("val" + flag + " sen", sen_mean)
    viz.plot("val" + flag + " fdr", fdr_mean)
    viz.plot("val" + flag + " spe", spe_mean)
    viz.plot("val" + flag + " bacc", bacc_mean)
    viz.plot("val" + flag + " kappa", kappa_mean)
    viz.plot("val" + flag + " gmean", gmean_mean)
    viz.plot("val" + flag + " iou", iou_mean)
    viz.plot("val" + flag + " dice", dice_mean)

    writer.add_scalars("val" + flag + "_loss", {"val" + flag + "_loss": loss_mean}, epoch)
    writer.add_scalars("val" + flag + "_auc", {"val" + flag + "_auc": auc_mean}, epoch)
    writer.add_scalars("val" + flag + "_acc", {"val" + flag + "_acc": acc_mean}, epoch)
    writer.add_scalars("val" + flag + "_sen", {"val" + flag + "_sen": sen_mean}, epoch)
    writer.add_scalars("val" + flag + "_fdr", {"val" + flag + "_fdr": fdr_mean}, epoch)
    writer.add_scalars("val" + flag + "_spe", {"val" + flag + "_spe": spe_mean}, epoch)
    writer.add_scalars("val" + flag + "_bacc", {"val" + flag + "_bacc": bacc_mean}, epoch)
    writer.add_scalars("val" + flag + "_kappa", {"val" + flag + "_kappa": kappa_mean}, epoch)
    writer.add_scalars("val" + flag + "_gmean", {"val" + flag + "_gmean": gmean_mean}, epoch)
    writer.add_scalars("val" + flag + "_iou", {"val" + flag + "_iou": iou_mean}, epoch)
    writer.add_scalars("val" + flag + "_dice", {"val" + flag + "_dice": dice_mean}, epoch)

    return dice_mean


def val(best_octa, viz, writer, dataloader, net,
        criterion, device, save_epoch_freq,
        models_dir, epoch, num_epochs=100):
    net.eval()
    loss_dct, auc_dct, acc_dct, sen_dct, fdr_dct, spe_dct, bacc_dct, kappa_dct, gmean_dct, iou_dct, dice_dct \
        = test(dataloader, net, device, criterion)
    dice = round(visual_results(loss_dct["octa"], auc_dct["octa"], acc_dct["octa"], sen_dct["octa"], fdr_dct["octa"],
                                spe_dct["octa"], bacc_dct["octa"], kappa_dct["octa"], gmean_dct["octa"], iou_dct["octa"],
                                dice_dct["octa"], viz, writer, epoch, flag="_octa") + 1e-12, 4)
    mkdir(models_dir)
    if dice >= best_octa["dice"]:
        best_octa["epoch"] = epoch + 1
        best_octa["dice"] = dice
        torch.save(net, os.path.join(models_dir, "model-best.pth"))

    print("best octa: epoch %d\tdice %.4f" % (best_octa["epoch"], best_octa["dice"]))

    '''checkpoint_path = os.path.join(models_dir, "{net}-{epoch}-{Dice}.pth")
    if (epoch + 1) % save_epoch_freq == 0:
        torch.save(net, checkpoint_path.format(net="model", epoch=epoch + 1, Dice=dice, ))
    if epoch == num_epochs - 1:
        torch.save(net, os.path.join(models_dir, "model-latest.pth"))
    net.train(mode=True)'''

    '''mkdir(models_dir)
    if auc >= best_octa["auc"]:
        best_octa["epoch"] = epoch + 1
        best_octa["auc"] = auc
        torch.save(net, os.path.join(models_dir, "model-best.pth"))

    print("best octa: epoch %d\tauc %.4f" % (best_octa["epoch"], best_octa["auc"]))'''

    '''checkpoint_path = os.path.join(models_dir, "{net}-{epoch}-{Auc}.pth")
    if (epoch + 1) % save_epoch_freq == 0:
        torch.save(net, checkpoint_path.format(net="model", epoch=epoch + 1, Auc=auc, ))
    if epoch == num_epochs - 1:
        torch.save(net, os.path.join(models_dir, "model-latest.pth"))
    net.train(mode=True)'''

    return net, best_octa
