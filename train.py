# -*- coding: utf-8 -*-
import torch
from torchvision.utils import save_image
from utils import mkdir, get_lr, adjust_lr


def train(viz, writer, dataloader, net, optimizer, base_lr, criterion, device, power,
          epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0

    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        mask = sample[1].to(device)
        # zero the parameter gradient
        optimizer.zero_grad()
        '''scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            pred = net(img)
            loss = criterion(pred, mask)
        scaler.scale(loss).backward()'''
        pred = net(img)
        loss = criterion(pred, mask)

        new = '%03d' % step
        name1 = 'D:/A_Project/OCTA-3M/results/img/' + new + ".png"
        name2 = 'D:/A_Project/OCTA-3M/results/mask/' + new + ".png"
        name3 = 'D:/A_Project/OCTA-3M/results/pred/' + new + ".png"

        viz.img(name="images", img_=img[0, :, :, :])
        viz.img(name="labels", img_=mask[0, :, :, :])
        viz.img(name="prediction", img_=pred[0, :, :, :])
        save_image(img[0, :, :, :], name1)
        save_image(mask[0, :, :, :], name2)
        save_image(pred[0, :, :, :], name3)

        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()  #
        optimizer.step()  #
        epoch_loss += loss.item()

        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        viz.plot("train loss", loss.item())

        # 写入当前lr
        current_lr = get_lr(optimizer)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)

    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)

    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)

    return net
