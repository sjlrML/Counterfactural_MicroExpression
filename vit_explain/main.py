import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
import shutil
sys.path.append("..") 
from vit_rollout2 import VITAttentionRollout
from vit_grad_rollout2 import VITAttentionGradRollout
from Trans_occ import occulsion_net
from model_Gaussian import Gaussian_model
from model_lstm import CNN_LSTM
from model_pca import PCA_model
from model_GCN import GCN_model

import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='/data/laolingjie/RAF-DB/96_128/test_0001.jpg',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.6,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_mask_on_image2(img,mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                img[i,j,:] = (0,0,255)
    return img


def add_forward(mask,feature):

    feature = feature.reshape(16,16)
    feature = cv2.resize(feature,(224,224))
    feature = feature - np.min(feature)
    feature = feature / np.max(feature)

    mask = mask + feature
    mask = np.maximum(mask,0)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    
    return mask

def read_image(path):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert('RGB').resize((224,224))
    x = data_transforms(image).unsqueeze(0)

    return image,x

def read_file(root,path):
    with open(os.path.join(path), 'r') as fd:
        lines = fd.readlines()
    labels = []
    apex_frame = []
    imgs = []
    for line in lines:
        line = line.strip()
        if ".jpg" in line:
            imgs.append(line)
        else:
            labels.append(line.split(" ")[1])
            apex_frame.append(line.split(" ")[0])

    return imgs,labels,apex_frame


if __name__ == '__main__':
    args = get_args()
    
    image_root = "/data/laolingjie/CASME2/Cropped/"
    train_list = "/data/laolingjie/code/Micro_FER/image_list.txt"
    save_dir = "/data/laolingjie/code/Micro_FER/heatmap/tmp/"
    
    backbone = occulsion_net()
    model = GCN_model(backbone,512,5)
    model.load_state_dict(torch.load("/data/laolingjie/code/Micro_FER/checkpoints/Occlusion_lstm_gcn_adj_add_feat2/Occlusion_lstm_gcn_adj_add_feat2_4_31_80.0_86.66666666666666.pth"))
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
        discard_ratio=args.discard_ratio)

    imgs,labels,apex_frames = read_file(image_root,train_list)
    count = 0

    for data in imgs:

        label = labels[int(count/9)]
        apex_frame = apex_frames[int(count/9)]

        image_path = data.strip()
        name = data.split('/')[-1].split('.jpg')[0]

        #read org image
        image,x = read_image(image_path)
        sub_dir = os.path.join(save_dir,image_path.split('/')[-3]+'/'+image_path.split('/')[-2]+'_'+label)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)        
        shutil.copy(image_path,sub_dir+'/'+str(name)+'.jpg')

        # pdb.set_trace()
        mask = attention_rollout([x.cuda()])
        np_img = np.array(image)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        
        # pdb.set_trace()    
        if apex_frame == name.split("reg_img")[1]:
            cv2.imwrite(sub_dir+'/'+str(name)+'_apex_vis.jpg',mask)
        else:
            cv2.imwrite(sub_dir+'/'+str(name)+'_vis.jpg',mask)
        count +=1

        if count %10==0:
            print("{}/100 has finished!!!!".format(count))



    print("finish!!!")

   

