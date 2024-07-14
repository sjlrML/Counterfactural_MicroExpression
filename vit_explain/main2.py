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
from model_pca2 import PCA_model

import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='/data/laolingjie/RAF-DB/96_128/test_0001.jpg',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.8,
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
    
    image_root = "/data/laolingjie/database/CASME2/Cropped/"
    train_list = "image_list_anger.txt"
    save_dir = "/data/laolingjie/code/Micro_FER/heatmap/tmp/"
    
    backbone = occulsion_net()
    model = PCA_model(backbone,768,5)
    model.load_state_dict(torch.load("../checkpoints/CoDer_SAMM/CoDer_SAMM_7_36_77.77777777777777_79.01234567901234.pth"))
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    imgs,labels,apex_frames = read_file(image_root,train_list)
    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
                                discard_ratio=args.discard_ratio)
    count = 0
    while count < len(imgs):

        label = labels[int(count/9)]
        apex_frame = apex_frames[int(count/9)]

        data = []
        images = []
        names = []
        for i in range(9):

            #read org image
            image_path = imgs[count].strip()
            name = imgs[count].split('/')[-1].split('.jpg')[0]
            image,x = read_image(image_path)
            sub_dir = os.path.join(save_dir,image_path.split('/')[-3]+'/'+image_path.split('/')[-2]+'_'+label)
            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)        
            shutil.copy(image_path,sub_dir+'/'+str(name)+'.jpg')
            data.append(x.cuda())
            images.append(image)
            names.append(name)
            count += 1

        mask = attention_rollout(data)
        for i in range(9):
            np_img = np.array(images[i])[:, :, ::-1]
            m = cv2.resize(mask[i], (np_img.shape[1], np_img.shape[0]))
            m = show_mask_on_image(np_img, m)
            
            # if apex_frame == names[i].split("reg_img")[1]:
            if apex_frame == names[i].split("_")[1]:
                cv2.imwrite(sub_dir+'/'+str(names[i])+'_apex_vis.jpg',m)
            else:
                cv2.imwrite(sub_dir+'/'+str(names[i])+'_vis.jpg',m)
 
        print("{}/100 has finished!!!!".format(count))



    print("finish!!!")

   

