import os
import torch
from skimage import morphology
from torchvision.utils import save_image
from tqdm import tqdm

from options import args
from PIL import Image
import numpy as np
from torchvision import transforms
import helper


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(args.models_dir + "/model-" + args.suffix).to(device)

img_folder = r'C:\Users\GCZX\Pictures\FAROS\faros_304\img'
result_folder = r'C:\Users\GCZX\Pictures\ex'

imglist = getFileList(img_folder, [], 'png')  # 注意输入图像格式

for imgpath in imglist:
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    img = Image.open(imgpath).convert("RGB")
    # img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img = np.array(img)
    img = img.transpose([2, 0, 1])
    img = helper.ImgToTensor(img)
    img = img.unsqueeze(0)
    pred = net(img)

    pred = pred.cpu().squeeze().detach().numpy().transpose([1, 2, 0])
    pred[pred < 0.25] = 0
    pred[pred > 0.25] = 1

    pred2 = pred[:, :, 0]
    pred1 = pred[:, :, 1]
    pred0 = pred[:, :, 2]

    pred1 = np.array(pred1, bool)
    pred1 = morphology.remove_small_objects(pred1, min_size=9, connectivity=1, in_place=True)
    pred1 = morphology.remove_small_holes(pred1, area_threshold=9, connectivity=2, in_place=True)
    pred1 = np.array(pred1 * 100, np.uint8)

    pred2 = np.array(pred2, bool)
    pred2 = morphology.remove_small_objects(pred2, min_size=100, connectivity=2, in_place=True)
    # pred2 = morphology.remove_small_holes(pred2, area_threshold=9, connectivity=2, in_place=True)
    pred2 = np.array(pred2 * 255, np.uint8)

    prediction = Image.fromarray(pred0 + pred1 + pred2).convert('L')
    prediction.save(result_folder + '/' + imgname + '.png')


    '''# single task segmentation
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    pred = pred.cpu().squeeze().detach().numpy()
    pred = np.array(pred, bool)
    pred = morphology.remove_small_objects(pred, min_size=16, connectivity=1, in_place=True)
    # img = morphology.remove_small_holes(img, area_threshold=16, connectivity=2, in_place=True)
    pred = np.array(pred * 255, np.uint8)
    prediction = Image.fromarray(pred).convert('L')
    prediction.save(result_folder + '/' + imgname + '.png')'''

    # save_image(pred[0, :, :, :], result_folder + '/' + imgname + '.png')
