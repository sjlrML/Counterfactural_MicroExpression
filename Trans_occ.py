import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.nn import LayerNorm
from models.Transformer import Part_Attention,Block
from models.Transformer import Transformer,CONFIGS, VisionTransformer
import pdb

config = CONFIGS["R50-ViT-B_16"]
transformer = VisionTransformer(config,224,zero_head=True,num_classes=7,vis=True)

class occulsion_net(nn.Module):
    def __init__(self):
        super(occulsion_net,self).__init__()
        
        self.num_ftrs = 768 * 1 * 1
        self.classes_num = 7
        self.res_block = Res_Block(transformer,config,224)
        self.trans = Trans_Block(transformer,config)
       
        # self.decoder = Decoder()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.classifier1 = nn.Linear(self.num_ftrs,7)

        
    def forward(self,image):
        image_feature_s1, image_feature_s2, image_feature_s3, image_feature_s4,image_feature_embedding = self.res_block(image)
        image_output,attn_weights,part_embedding,part_inx,part_weights = self.trans(image_feature_embedding)
# 
        return part_embedding[:,0],attn_weights
        # return image_feature_embedding,None


class Res_Block(nn.Module):
    def __init__(self,transformer_res,config,img_size,in_channels=3):
        super(Res_Block,self).__init__()
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        n_patches = (img_size[0] // 16) * (img_size[1] // 16)

        self.head = transformer_res.transformer.embeddings.hybrid_model.root
        self.res_block1 = transformer_res.transformer.embeddings.hybrid_model.body.block1
        self.res_block2 = transformer_res.transformer.embeddings.hybrid_model.body.block2
        self.res_block3 = transformer_res.transformer.embeddings.hybrid_model.body.block3
        self.dropout = transformer_res.transformer.embeddings.dropout
        self.path_embeddings = transformer_res.transformer.embeddings.patch_embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self,x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        output1 = self.head(x)
        output2 = self.res_block1(output1)
        output3 = self.res_block2(output2)
        output4 = self.res_block3(output3)
        x = self.dropout(output4)
        x = self.path_embeddings(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)


        return output1,output2,output3,output4,embeddings

class Trans_Block(nn.Module):
    def __init__(self,transformer_res,config):
        super(Trans_Block,self).__init__()
        self.trans_block = nn.Sequential(*list(transformer_res.transformer.encoder.layer[0:4]))
        self.part_select = Part_Attention()
        self.part_layer = Block(config,vis=True)
        self.part_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)


    def forward(self, x):
        attn_weights = []
        for layer_block in self.trans_block:
            x, weights = layer_block(x)
            attn_weights.append(weights)
        featrue = x

        part_num, part_inx = self.part_select(attn_weights)
        part_inx = part_inx + 1
        parts = []
        B, num = part_inx.shape
        for i in range(B):
            parts.append(x[i, part_inx[i,:]])
        parts = torch.stack(parts).squeeze(1)
        concat = torch.cat((x[:,0].unsqueeze(1), parts), dim=1)
        part_states, part_weights = self.part_layer(concat)
        part_encoded = self.part_norm(part_states)
                
        return featrue, attn_weights,part_encoded,part_inx,part_weights
        # return featrue, attn_weights

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.conv = BasicConv(768,1024,kernel_size=1)
        self.up_sample1 = BasicConv_upsample(1024,512)
        self.up_sample2 = BasicConv_upsample(512,256,kernel_size=2)
        self.up_sample3 = BasicConv_upsample(256,64,stride=1,kernel_size=1,output_padding=0,padding=0)
        self.up_sample4 = BasicConv_upsample(64,64,kernel_size=5)
        self.up_sample5 = BasicConv_upsample(64,3)
        


        
        # self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)


    def forward(self,x,occ_feature_s1,occ_feature_s2,occ_feature_s3,occ_feature_s4):
        # x = self.conv(x) + self.alpha * occ_feature_s4
        # x = self.up_sample1(x) + self.alpha * occ_feature_s3
        # x = self.up_sample2(x) + self.alpha * occ_feature_s2
        # x = self.up_sample3(x) + self.alpha * occ_feature_s1
        # x = self.up_sample4(x)
        # x = self.up_sample5(x)

        x = self.conv(x) +  occ_feature_s4
        x = self.up_sample1(x) +  occ_feature_s3
        x = self.up_sample2(x) +  occ_feature_s2
        x = self.up_sample3(x) +  occ_feature_s1
        x = self.up_sample4(x)
        x = self.up_sample5(x)


        return x

class BasicConv_upsample(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=2,padding=1,output_padding=1):

        super(BasicConv_upsample, self).__init__()
        self.Conv_BN_ReLU = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        x_out=self.Conv_BN_ReLU(x)
        x_out=self.upsample(x_out)
        return x_out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




