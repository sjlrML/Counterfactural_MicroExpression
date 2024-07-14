import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from models.resnet import resnet50,resnet18
from models.vgg import vgg16
from sklearn.metrics.pairwise import cosine_similarity
# from audtorch.metrics.functional import pearsonr
import pdb


class PCA_model(nn.Module):
    def __init__(self,backbone,feature_size,classes_num,frames_num = 9,LSTM_layer=1):
        super(PCA_model,self).__init__()
        
        self.classes_num = classes_num
        self.feature_extractor = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    
            
        # self.projection = Parameter(torch.FloatTensor(768,768).cuda())
        # self.projection = nn.Linear(768,768,bias=False)

        self.lstm = nn.LSTM(input_size=768, hidden_size=feature_size, num_layers=LSTM_layer, batch_first=True, bidirectional=False)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size,feature_size//2),
            nn.BatchNorm1d(feature_size//2),
            nn.ELU(inplace=True),
            nn.Linear(feature_size//2, classes_num)
        )  

    def forward(self,images,frame_index=None,apex_index=None,phase="train"):
        features = []
        logtis = []
        logtis_aug = []

        attns = []
        for image in images:
            x,attention = self.feature_extractor(image)
            # x = torch.mean(x,1)
            features.append(x)
            attns.append(attention)
        
        features = torch.stack(features,dim=1)
        B,T,C = features.shape

        # # eigenvalue decomposition
        # new_features = []
        # mean_feature = torch.mean(features,dim=2)
        # nor_feature = features - mean_feature.unsqueeze(2)
        # cov_matrix = torch.matmul(nor_feature,nor_feature.permute(0,2,1))
        # for i in range(cov_matrix.size(0)):
        #     e,v = torch.eig(cov_matrix[i],eigenvectors=True)
        #     new_features.append(torch.mm(v,features[i]))  

        # new_features = torch.stack(new_features) + features

        # new_features = new_features.permute(1,0,2)


        L_features,_ = self.lstm(features)
        L_features = L_features.permute(1,0,2)
        
        for i,feature in enumerate(L_features):
            logtis.append(self.classifier(feature))

        logtis = torch.stack(logtis,dim=1)
        outputs = torch.sum(logtis,dim=1)

        if phase == "test":
            return outputs,attns
        
        return  outputs,0,0,0

        # s = 1
        # new_features_aug = L_features.clone()
        # for i in range(B):
        #     new_features_aug[apex_index[i]-s:apex_index[i]+s+1,i,:] = new_features_aug[0,i]
        
        # s = 6
        # new_features_aug = L_features.clone()
        # rand_index = torch.randperm(s)
        # new_features_aug = new_features_aug[rand_index]
        
        # rand_num = torch.rand(1)
        # if rand_num > 0.05:
        # s = 1
        # new_features_aug = new_features.clone().detach()
        # for i in range(B):
        #     new_features_aug[apex_index[i]-s:apex_index[i]+s+1,i,:] = new_features_aug[0,i]
        # else:
        s = 6
        rand_index = torch.randperm(s)
        new_features_aug = L_features.clone()
        new_features_aug = new_features_aug[rand_index]



        # new_features_aug = new_features[:,torch.randperm(new_features.size(1))]
        # L_features, L_weight= self.lstm(new_features_aug)
        # L_features = L_features.permute(1,0,2)
        
        for i,feature in enumerate(new_features_aug):
            logtis_aug.append(self.classifier(feature))

        logtis_aug = torch.stack(logtis_aug,dim=1)
        outputs_aug = torch.sum(logtis_aug,dim=1)

        
        return  outputs,outputs - outputs_aug,0,0
        # return  outputs,0,0,0


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

def adj_normalize(adj):
    D = torch.pow(adj.sum(1).float(),-0.5)
    D = torch.diag(D)
    adj_nor = torch.matmul(torch.matmul(adj,D).t(),D)
    return adj_nor

# def adjacent_matrix_generator(batch_size,node_num):

#     adj = torch.ones((batch_size,node_num,node_num),requires_grad=True)
#     for i in range(batch_size):
#         adj[i] = adj_normalize(adj[i])

#     return adj.float().cuda()

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def cos_sim(a,b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        return cosine_similarity(a,b)
    elif a.ndim==2:
        dis = np.zeros((a.shape[0],a.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                dis[i,j] = cosine_similarity(a[i],b[j])
        return dis

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
