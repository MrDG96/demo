# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
# import cv2
import random

import numpy
import torch
from PIL import Image
import numpy as np
import helper
import joint_transforms
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF

palette = [[255], [100], [0]]
num_classes = 3


# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(512, 512)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)

    return image, label


class octa(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(octa, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""
        self.palette = palette

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        img = Image.open(imgPath)
        mask = Image.open(maskPath)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = np.array(img)
        mask = np.array(mask)

        # img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helper.mask_to_onehot(mask, self.palette)

        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])

        img = helper.ImgToTensor(img)
        mask = helper.MaskToTensor(mask)

        '''img = transforms.ToTensor(img)
        mask = transforms.ToTensor(mask)'''

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class faros(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(faros, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""
        self.palette = palette

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        img = Image.open(imgPath)
        mask = Image.open(maskPath)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = np.array(img)
        mask = np.array(mask)

        # img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helper.mask_to_onehot(mask, self.palette)

        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])

        img = helper.ImgToTensor(img)
        mask = helper.MaskToTensor(mask)

        '''img = transforms.ToTensor(img)
        mask = transforms.ToTensor(mask)'''

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class octa1(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(octa1, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        simple_transform = transforms.ToTensor()

        img = Image.open(imgPath)
        mask = Image.open(maskPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        mask = np.array(mask)
        mask[mask >= 128] = 255
        mask[mask < 128] = 0
        mask = Image.fromarray(mask)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        img = simple_transform(img)
        mask = simple_transform(mask)

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class octa2(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(octa2, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        simple_transform = transforms.ToTensor()

        img = Image.open(imgPath)
        mask = Image.open(maskPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        mask = np.array(mask)
        mask[mask == 255] = 0
        mask[mask == 100] = 255
        mask = Image.fromarray(mask)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        img = simple_transform(img)
        mask = simple_transform(mask)

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class neural(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(neural, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""
        self.palette = palette

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        img = Image.open(imgPath)
        mask = Image.open(maskPath)

        joint_transform = joint_transforms.Compose([joint_transforms.RandomScaleCrop()])

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)
            # img, mask = joint_transform(img, mask)

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = np.array(img)
        mask = np.array(mask)

        # img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helper.mask_to_onehot(mask, self.palette)

        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])

        img = helper.ImgToTensor(img)
        mask = helper.MaskToTensor(mask)

        '''img = transforms.ToTensor(img)
        mask = transforms.ToTensor(mask)'''

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class neural1(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(neural1, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        simple_transform = transforms.ToTensor()

        img = Image.open(imgPath)
        mask = Image.open(maskPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        mask = np.array(mask)
        mask[mask >= 100] = 255
        mask[mask < 100] = 0
        mask = Image.fromarray(mask)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        img = simple_transform(img)
        mask = simple_transform(mask)

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class vessel(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(vessel, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        simple_transform = transforms.ToTensor()

        img = Image.open(imgPath)
        mask = Image.open(maskPath).convert("L")

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        mask = np.array(mask)
        mask[mask >= 128] = 255
        mask[mask < 128] = 0
        mask = Image.fromarray(mask)

        if self.isTraining:
            # augmentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        img = simple_transform(img)
        mask = simple_transform(mask)

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name


class vessel2(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, isTesting=False):
        super(vessel2, self).__init__()
        self.img, self.mask = self.get_dataPath(root, isTraining, isTesting)
        self.channel = channel
        self.isTraining = isTraining
        self.isTesting = isTesting
        self.name = ""
        self.palette = palette

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img[index]
        self.name = imgPath.split("/")[-1]
        maskPath = self.mask[index]

        img = Image.open(imgPath)
        mask = Image.open(maskPath)

        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            mask = mask.rotate(angel)

        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = np.array(img)
        mask = np.array(mask)

        # img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helper.mask_to_onehot(mask, self.palette)

        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])

        img = helper.ImgToTensor(img)
        mask = helper.MaskToTensor(mask)

        return img, mask

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img)

    def get_dataPath(self, root, isTraining, isTesting):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            mask_dir = os.path.join(root + "/train/mask")
        elif isTesting:
            img_dir = os.path.join(root + "/test/img")
            mask_dir = os.path.join(root + "/test/mask")
        else:
            img_dir = os.path.join(root + "/validation/img")
            mask_dir = os.path.join(root + "/validation/mask")

        img = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        mask = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))

        return img, mask

    def getFileName(self):
        return self.name
