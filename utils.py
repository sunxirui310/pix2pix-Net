"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from PIL import Image

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False,is_random_rot=False,load_size=286,fine_size=256):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test,is_random_rot=is_random_rot,load_size=load_size,fine_size=fine_size)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False,is_random_rot=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if is_random_rot:
            angle=np.random.random_sample()*360-180
            img_A = rot_image(img_A,fine_size,angle)
            img_B = rot_image(img_B,fine_size,angle)

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

def rot_image(im_arr,fine_size,angle):
    src_im = Image.fromarray(im_arr)
    

    src_im_Rs=src_im
    dst_im = Image.new("RGB", (fine_size,fine_size) )
    im = src_im_Rs.convert('RGB')
    rot = im.rotate( angle, expand=1 ).resize((fine_size,fine_size))
    dst_im.paste( rot )
    rot_array=np.array(dst_im)
    return rot_array
# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def load_blind_data(image_path, flip=True, is_test=False,load_size=286,fine_size=256):
    input_img = imread(image_path)
    ori_h,ori_w,channel=input_img.shape
    input_img_rsh=scipy.misc.imresize(input_img,[fine_size,fine_size]).astype(np.float)
    
    img_A = np.zeros((fine_size,fine_size,3))
    img_B = input_img_rsh

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


def load_blind_datasize(image_path, flip=True, is_test=False):
    input_img = imread(image_path)
    ori_h,ori_w,channel=input_img.shape
    
    return (ori_h,ori_w)

def resize_back(im,rows,cols,tobw=True,thr=80):
   
    image=scipy.misc.imresize((im+1.)/2.,(rows,cols),interp="bicubic")

    if tobw:
        bw=np.where(image>thr,255,0)
    else:
        bw=image
    return bw

