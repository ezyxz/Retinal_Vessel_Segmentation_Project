#coding:utf8

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import operator
from PIL import Image
import sys
from config_DRIVE import *

def is_image_file(filename):  # 
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif",".tiff"])

def is_image_damaged(cvimage):
    white_pixel_count = 0
    height,weight,channel = cvimage.shape
    # for row in list(range(0,height)):
    #     for col in list(range(0,weight)):
    #         if cvimage[row][col][2] == 255:
    #             white_pixel_count += 1
    #             if white_pixel_count > 0.2*height*weight:
    #                 return True
    # return False
    one_channel = np.sum(cvimage, axis=2)
    white_pixel_count = len(one_channel[one_channel==255*3])   #Count the number of white pixels
    if white_pixel_count > 0.08*height*weight:
        return True
    return False

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    
def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：Flip along the y axis
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_sets,img_w,img_h,image_num = 5000, mode = 'normal'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread(src_data_path + image_sets[i])  # 3 channels
        # label_img = cv2.imread('./Training Set/Target maps/' + image_sets[i].replace('tiff','tif'))  # 3 channels
        lab_path = label_data_path + image_sets[i].split('_')[0]+'_manual1.gif'
        assert os.path.exists(lab_path)
        # label_img_gray = cv2.imread(lab_path,0)  # single channel
        label_img_gray = np.asarray(Image.open(lab_path))
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            # if is_image_damaged(src_roi):
            #     continue
            # label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi_gray = label_img_gray[random_height: random_height + img_h, random_width: random_width + img_w]
            if np.max(label_roi_gray)>0:
                if mode == 'augment':
                    src_roi, label_roi_gray = data_augment(src_roi, label_roi_gray)
                if dataset == 'DRIVE':
                    label_roi_gray = np.where(label_roi_gray > 128, 1, 0)

                # cv2.imwrite(('./Training Set/visualize/%d.png' % g_count),label_roi)
                cv2.imwrite(os.path.join(target_path, 'image_train', '%d.png' % g_count), src_roi)
                cv2.imwrite(os.path.join(target_path, 'label_train', '%d.png' % g_count), label_roi_gray)
                count += 1
                g_count += 1

if __name__ == '__main__':
    img_w = 512
    img_h = 512
    image_num = 4000
    dataset = dataset
    src_data_path = './{}_new/big_img/'.format(dataset)
    label_data_path = './{}_new/big_gt/'.format(dataset)
    target_path = './data_slice_{}'.format(dataset)
    assert os.path.exists(target_path) is False
    if os.path.exists(target_path) is False:
        os.mkdir(target_path)
        os.mkdir(os.path.join(target_path,'image_train'))
        os.mkdir(os.path.join(target_path,'label_train'))

    image_sets2 = [x for x in os.listdir(src_data_path) if is_image_file(x)]
    creat_dataset(image_sets=image_sets2,img_w=img_w,img_h=img_h, image_num = image_num)
