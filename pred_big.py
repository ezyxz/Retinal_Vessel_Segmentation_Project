# -*- coding:utf-8 -*-
import cv2
import csv
import tqdm
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable
import os
import transform as tr
from networks import fpn_unet

np.seterr(divide='ignore', invalid='ignore')
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import shutil

from cal_iou import evaluate
from config_CHASE import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    # img, label = random_crop(img, label, crop_size)
    transform = transforms.Compose([
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']

def find_new_file(dir):
    if os.path.exists(dir) is False:
        os.mkdir(dir)
        dir = dir

    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None

def label_mapping(label_im):
    # colorize = np.zeros([2, 3], dtype=np.int64)
    colorize = np.array([[0, 0, 0],
                [255, 255, 255],
                [0, 255, 255],
                [0, 255, 0],
                [0, 0, 255],
                [128, 128, 0],
                [192, 128, 128],
                [64, 64, 128],
                [64, 0, 128],
                [64, 64, 0],
                [0, 128, 192],
                [255, 0, 0]
                ])
    # colorize[0, :] = [128, 128, 0]
    # colorize[1, :] = [255, 255, 255]
    label = colorize[label_im, :].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label

def predict(net, im): # 预测结果
    # use_gpu = False
    with torch.no_grad():
        if use_gpu:
            im = im.unsqueeze(0).cuda()
        else:
            im = im.unsqueeze(0)
        output = net(im)
        pred = output.max(1)[1].squeeze().cpu().data.numpy()
        pred_ = label_mapping(pred)
    return pred_, pred

def pred_image(net, img_, output, output_gray, img_path):
    img = Image.open(os.path.join(img_)).convert('RGB')
    shape = img.size
    img_new = img.resize((shape[0] * 5, shape[1] * 5))
    imsw, imsh = img_new.size
    crop_size = img_size

    xw = int(imsw / crop_size)
    xh = int(imsh / crop_size)

    new_size = [(xw + 1) * crop_size, (xh + 1) * crop_size]
    new_img = img_new.resize((new_size[0], new_size[1]), Image.ANTIALIAS)
    pred = np.zeros((new_size[1], new_size[0], 3))
    all_gray = np.zeros((new_size[1], new_size[0]))
    for i in range(xh + 1):
        for j in range(xw + 1):
            img2 = new_img.crop((crop_size * j, crop_size * i, crop_size * (j + 1), crop_size * (i + 1)))  # hengzhede
            name = img_path + str(i) + '_src_' + str(j) + '.png'
            img2.save(name)
            image = Image.open(name)
            image_np = img_transforms(image)
            res, gray = predict(net, Variable(torch.Tensor(image_np)).cuda())
            # res, gray = predict(net, Variable(torch.Tensor(image_np)))
            im1 = Image.fromarray(np.uint8(res))
            im1.save((name.replace('src', 'pred')))
            im2 = Image.fromarray(np.uint8(gray))
            im2.save((name.replace('src', 'gray')))
            pred[crop_size * i:crop_size * (i + 1), crop_size * j:crop_size * (j + 1)] = res
            all_gray[crop_size * i:crop_size * (i + 1), crop_size * j:crop_size * (j + 1)] = gray

    result_img = Image.fromarray(np.uint8(pred))
    result_img = result_img.resize((shape[0], shape[1]))
    result_img.save(output + img_.split('/')[-1])
    result_img_gray = Image.fromarray(np.uint8(all_gray))
    result_img_gray = result_img_gray.resize((shape[0], shape[1]))
    result_img_gray.save(output_gray + img_.split('/')[-1])

def dig_img(output_gray, src, tgt):
    imgs = os.listdir(output_gray)

    for i in range(len(imgs)):
        print(imgs[i])
        img1 = cv2.imread(os.path.join(output_gray, imgs[i]), 0)
        img2 = cv2.imread(os.path.join(src, imgs[i]))
        size = list(np.shape(img1))
        img3 = np.zeros((size[0], size[1], 3))

        for j in range(size[0]):
            for k in range(size[1]):
                if img1[j][k]==1:
                    img3[j][k]=img2[j][k]
        cv2.imwrite(os.path.join(tgt, imgs[i]), img3)


def test_my(input_bands, model_name, model_dir, img_size, num_class):
    net = fpn_unet(input_bands=input_bands, n_classes=num_class)
    imgs = os.listdir(val_path)
    if os.path.exists(output):
        shutil.rmtree(output)
        os.mkdir(output)
    else:
        os.mkdir(output)
    if os.path.exists(output_gray):
        shutil.rmtree(output_gray)
        os.mkdir(output_gray)
    else:
        os.mkdir(output_gray)
    state_dict = torch.load(find_new_file(model_dir))
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)
    net.load_state_dict(state_dict)

    if use_gpu:
        net.cuda()
    net.eval()
    for i in tqdm.tqdm(range(len(imgs))):
        img_path = output + imgs[i][0:-4] + '/'
        os.mkdir(img_path)
        pred_image(net, os.path.join(val_path, imgs[i]), output, output_gray, img_path)
    iou, acc, recall, precision = evaluate(output_gray, val_gt, num_class)
    return iou, acc, recall, precision

# x, y, recall, precision = test_my(input_bands, model_name, model_dir, img_size, num_class)
# print([x, y, recall, precision])