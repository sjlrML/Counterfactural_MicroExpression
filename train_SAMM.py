import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np
import random
import time
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.optim as optim
from dataset import CASME2,SAMM
from torch.utils import data
from models.resnet import resnet50,resnet18
from models.tcae_model import FrontaliseModelMasks_wider
from Trans_occ import occulsion_net
from models.I3D import InceptionI3d
from model_base import Base_model
from model_lstm import CNN_LSTM
from model_I3D import I3D
from model_vac import VAC_model
from model_Gaussian2 import Gaussian_model
from model_GCN import GCN_model
from model_pca2 import PCA_model
from model_pca_gcn2 import PCA_GCN_model
from dataset_config import CASME2_config,SAMM_config
import pickle
import random
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_args():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--data_root',type=str,default='/data/laolingjie/database/SAMM/SAMM')
    parse.add_argument('--data_root',type=str,default='/data/laolingjie/database/SAMM/Cropped')
    parse.add_argument('--data_list',type=str,default='/data/laolingjie/database/SAMM/SAMM_Micro_FACS_Codes_v2_my.xlsx')
    parse.add_argument('--checkpoint',type=str,default='checkpoints/')
    parse.add_argument('--resume',type=bool,default=False)
    parse.add_argument('--model_path',type=str,default='')
    parse.add_argument('--train_log',type=str,default='log/SAMM_baseline.log')
    parse.add_argument('--save_name',type=str,default='SAMM_baseline')
    parse.add_argument('--start_epoch',type=int,default=0)
    parse.add_argument('--nb_epoch',type=int,default=80)
    parse.add_argument('--batch_size',type=int,default=12)
    parse.add_argument('--num_workers',type=int,default=8)
    parse.add_argument('--classes_num',type=int,default=5)
    parse.add_argument('--lstm_num',type=int,default=1)
    parse.add_argument('--sample_method',type=str,default="interval",help="interval or apex")
    parse.add_argument('--alpha',type=float,default=0.1,help='the loss weight of Gaussian loss')

    return parse.parse_args()

def save_model(model, save_path, name, iter_cnt,test_subject_index,acc,f1):  
    if not os.path.exists(os.path.join(save_path,name)):
        os.mkdir(os.path.join(save_path,name))                                                                                 
    save_name = os.path.join(os.path.join(save_path,name), name + '_' + str(iter_cnt)+ '_' + str(test_subject_index) + '_'+ str(acc)+ '_' + str(f1) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)

def load_vgg_pretrained_pkl(path):
    with open(path, 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    return weights

def val(model,val_loader,CELoss,args,epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    total_pred = []
    total_label = []
    with torch.no_grad():
        for batch_idx,(image_list,labels) in enumerate(val_loader):
            idx = batch_idx
            num = int(len(image_list)/9)
            best_acc = 0

            labels = labels.long().cuda()
            image_list = [img.cuda() for img in image_list]
            total += labels.size(0)
            outputs = []
            for i in range(num):
                images = image_list[9*i:9*(i+1)]
                output,_ = model(images,phase="test")
                outputs.append(output)

            outputs = torch.stack(outputs)
            N = outputs.shape[0]
            outputs = torch.sum(outputs,dim=0)/N
            loss = CELoss(outputs,labels)
            val_loss +=loss.item()

            #prediction
            _,predicted = torch.max(outputs.data,1)   
            correct += predicted.eq(labels.data).cpu().sum() 
            total_pred.append(predicted.detach().cpu().numpy().tolist())
            total_label.append(labels.detach().cpu().numpy().tolist())
            

    val_acc = 100. *float(correct) /total
    val_loss = val_loss /(idx+1)
    total_label = np.concatenate(total_label)
    total_pred =  np.concatenate(total_pred)
    f1 = f1_score(total_label,total_pred,average='weighted')*100

    print(
        'Iteration %d | val_acc = %.5f | val_F1 = %.5f | val_loss = %.5f |' % (
            epoch,val_acc,f1,val_loss))
    print('--'*40)
    with open(args.train_log,'a') as file:
        file.write(
            'Iteration %d | val_acc = %.5f | val_F1 = %.5f | val_loss = %.5f |' % (
                epoch,val_acc,f1,val_loss))
        file.write('--'*40+'\n')
    
    return val_acc,f1,correct,total,list(total_label),list(total_pred)
    

def train(train_loader,model,CELoss,optimizer,epoch,args):
    train_loss = 0
    con_loss = 0
    ce_loss = 0
    correct = 0
    total = 0
    fold_pred = []
    fold_label = []
    model.train()
    # KLLoss = nn.KLDivLoss(reduction="batchmean")
    margin = 5e-1
    for batch_idx,(image_list,labels,image_index_list,apex_index) in enumerate(train_loader): 
        labels = labels.long().cuda()
        image_list = [img.cuda() for img in image_list]

        optimizer.zero_grad()
        outputs,outputs_effect,_,_ = model(image_list,image_index_list,apex_index)

        loss4 = CELoss(outputs,labels)
        # loss1 =  CELoss(outputs_effect,labels)
        # loss = loss4 + args.alpha * (loss1)
        loss = loss4
        loss.backward()
        optimizer.step()

        # con_loss += loss1.item()
        ce_loss += loss4.item()
        train_loss += loss.item()

        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        fold_pred.append(predicted.detach().cpu().numpy().tolist())
        fold_label.append(labels.detach().cpu().numpy().tolist())
        
        if batch_idx % 5 == 0:
            print("Sample: {}/{} Iteration {} Step: {} | Loss: {:.3f} | Acc: {:.3f} ({}/{}) |" \
                 "Counterfactual Loss: {:.3f} | CE Loss: {:.3f} |".format(
                i_fold+1,len(subject_index),epoch,batch_idx,train_loss / (batch_idx + 1),100. *float(correct)/total,correct,total,
                args.alpha *con_loss / (batch_idx+1),ce_loss / (batch_idx+1)))
            with open(args.train_log,'a') as file:
                file.write("Sample: {}/{} Iteration {} Step: {} | Loss: {:.3f} | Acc: {:.3f} ({}/{}) |" \
                 "Counterfactual Loss: {:.3f} | CE Loss: {:.3f} |\n".format(
                i_fold+1,len(subject_index),epoch,batch_idx,train_loss / (batch_idx + 1),100. *float(correct)/total,correct,total,
                args.alpha *con_loss / (batch_idx+1), ce_loss / (batch_idx+1)))

    fold_label = np.concatenate(fold_label)
    fold_pred =  np.concatenate(fold_pred)
    f1 = f1_score(fold_label,fold_pred,average='micro')*100

    train_acc = 100. *float(correct) /total
    print(
        'Iteration %d Sample: %d/%d | Acc = %.5f | F1_socre = %.5f | train_loss = %.5f |' % (
            epoch,i_fold+1,len(subject_index),train_acc,f1,train_loss/ (batch_idx + 1)))
    with open(args.train_log,'a') as file:
        file.write(
            'Iteration %d Sample: %d/%d | Acc = %.5f | F1_socre = %.5f | train_loss = %.5f |\n' % (
            epoch,i_fold+1,len(subject_index),train_acc,f1,train_loss/ (batch_idx + 1)))


if __name__ == "__main__":
    args = get_args()
    # setup_seed(20)
    setup_seed(4047)
    print(args)

    # subject_index = [i for i in range(1,CASME2_config["subject_num"]+1)]
    subject_index = SAMM_config["subject_num"]
    random.shuffle(subject_index)
    best_acc = []
    best_f1 = []
    best_correct = 0
    best_total = 0
    best_label = []
    best_pred = []
    for i_fold,test_subject_index in enumerate(subject_index):
        print("*****fold {}/{}*****".format(i_fold+1,len(subject_index)))

        train_dataset = SAMM(args.data_root,args.data_list, class_num = args.classes_num,test_subject=test_subject_index,is_dynamic=False,
                        sample=args.sample_method,phase='train',input_shape=(3,224,224))
        train_loader = data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args.num_workers)

        val_dataset = SAMM(args.data_root,args.data_list,class_num = args.classes_num,test_subject=test_subject_index,is_dynamic=False,
                                sample=args.sample_method,phase='test',input_shape=(3,224,224))
        val_loader = data.DataLoader(val_dataset,
                                    batch_size=4,
                                    num_workers=2)
        
        # ***** TCAE backbone ***** #
        # backbone = FrontaliseModelMasks_wider(inner_nc=256, \
        #     num_output_channels=3, num_masks=0, num_additional_ids=32)
        # state_dict = torch.load("/laolingjie/code/checkpoints/tcae_epoch_1000.pth")
        # backbone.load_state_dict(state_dict["state_dict"])
        # backbone = backbone.encoder

        # ***** ResNet50 backbone pretrained on VGGFace ***** # 
        # backbone = resnet50(pretrained=True)
        # model_state = backbone.state_dict()
        # past_state = load_vgg_pretrained_pkl("/laolingjie/code/checkpoints/resnet50_ft_weight.pkl")
        # pretrained_dict = {k : v for k, v in past_state.items() if k in model_state}
        # for k,v in list(pretrained_dict.items()):
        #     if 'fc' in k:
        #         pretrained_dict.pop(k)
        # model_state.update(pretrained_dict)
        # backbone.load_state_dict(model_state)

        # ***** I3D backbone **** #
        # backbone = InceptionI3d(num_classes=400,in_channels=3)

        # ***** Occlusion Transformer backbone **** #
        backbone = occulsion_net()
        # backbone.load_state_dict(torch.load("/data/laolingjie/code/checkpoints/occulsion_net_trans_4layer_more_113_86.17992177314211_76.7978159604637.pth"),strict=False)
        # backbone.load_state_dict(torch.load("/data/laolingjie/code/checkpoints/occulsion_net_trans_no_gen_cls_loss_part_select_58_88.46153846153847_81.35203156083615.pth"),strict=False)

        # model = Base_model(backbone,512,args.classes_num)
        # model = Gaussian_model(backbone,256,args.classes_num,LSTM_layer=args.lstm_num)
        # model = GCN_model(backbone,512,args.classes_num)
        # model = PCA_GCN_model(backbone,512,args.classes_num)
        model = PCA_model(backbone,768,args.classes_num)
        # model = VAC_model(backbone,512,args.classes_num,LSTM_layer=args.lstm_num)
        # model = I3D(backbone,512,args.classes_num)
        
        if i_fold == 0:
        	print(model)


        if args.resume:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
        model.cuda()
        model = torch.nn.DataParallel(model)

        CELoss = nn.CrossEntropyLoss()
        optimizer = optim.SGD([
            {'params': model.module.feature_extractor.parameters(),'lr':0.0003},
            {'params': model.module.classifier.parameters(),'lr':0.003},
            # {'params': model.module.lstm.parameters(),'lr':0.001},
        ],
            momentum=0.9,weight_decay=5e-4)

        # lr = [0.0005,0.005,0.005,0.005,0.005,0.005,0.005]
        lr = [0.0003,0.003,0.003,0.003,0.003]
        # lr = [0.0002,0.002,0.002,0.002,0.002,0.002,0.002]
        # lr = [0.0001,0.001,0.001,0.001,0.001,0.001,0.001]

        epoch_acc = 0.0
        epoch_f1 = 0.0
        epoch_correct = 0
        epoch_total= 0
        for epoch in range(args.start_epoch+1,args.nb_epoch+1):
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.nb_epoch, lr[nlr])
            
            train(train_loader,model,CELoss,optimizer,epoch,args)
            acc,f1,c,t,l,p = val(model,val_loader,CELoss,args,epoch)
            if acc > epoch_acc:
                epoch_acc = acc
                epoch_correct = c
                epoch_total = t
                save_model(model.module,'checkpoints',args.save_name,test_subject_index,epoch,acc,f1)
                np.save("checkpoints/"+str(args.save_name)+"/"+str(test_subject_index)+"_"+str(epoch)+"_"+str(acc)+"_"+str(f1)+"_label",np.array(l))
                np.save("checkpoints/"+str(args.save_name)+"/"+str(test_subject_index)+"_"+str(epoch)+"_"+str(acc)+"_"+str(f1)+"_predicted",np.array(p))
            if f1 > epoch_f1:
                epoch_f1 = f1
                epoch_label = l
                epoch_pred = p
                if f1 >=55:
                    save_model(model.module,'checkpoints',args.save_name,test_subject_index,epoch,acc,f1)

            time_str = time.asctime(time.localtime(time.time()))
            print('{} Sample: {}/{} Iteration {} finish!!!'.format(time_str,i_fold+1,len(subject_index),epoch))
            with open(args.train_log,'a') as file:
                file.write('{} Sample: {}/{} Iteration {} finish!!!\n'.format(time_str,i_fold+1,len(subject_index),epoch))
            if acc > 99.0:
                break
        best_acc.append(epoch_acc)
        best_f1.append(epoch_f1)
        best_correct += epoch_correct
        best_total += epoch_total
        best_pred.extend(epoch_pred)
        best_label.extend(epoch_label)
        socre = f1_score(best_label,best_pred,average='micro')*100
        print(
            'Sample {}/{} | Avg Acc = {:.5f} ({})/({}) | Avg F1_socre = {:.5f} |'.format(i_fold+1,len(subject_index),best_correct/best_total*100,best_correct,best_total,socre))
        with open(args.train_log,'a') as file:
            file.write(
                'Sample {}/{} | Avg Acc = {:.5f} | Avg F1_socre = {:.5f} |\n'.format(i_fold+1,len(subject_index),best_correct/best_total*100,best_correct,best_total,socre))

        # print(
        #     'Sample {}/{} | Avg Acc = {:.5f} | Avg F1_socre = {:.5f} |'.format(i_fold+1,len(subject_index),np.mean(best_acc),np.mean(best_f1)))
        # with open(args.train_log,'a') as file:
        #     file.write(
        #         'Sample {}/{} | Avg Acc = {:.5f} | Avg F1_socre = {:.5f} |\n'.format(i_fold+1,len(subject_index),np.mean(best_acc),np.mean(best_f1)))
    avg_acc = np.mean(best_acc)
    avg_f1 = np.mean(best_f1)
    print(
        'Acc = %.5f | F1_socre = %.5f |' % (avg_acc,avg_f1))
    with open(args.train_log,'a') as file:
        file.write(
            'Acc = %.5f | F1_socre = %.5f |\n' % (avg_acc,avg_f1))














