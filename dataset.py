import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pandas as pd
from PIL import Image
import cv2
import pdb
# from dynamicimage import get_dynamic_image



CASME2_label_to_cls_7 ={"happiness":0,"others":1,"disgust":2,"surprise":3,"fear":4,"repression":5,"sadness":6}
CASME2_label_to_cls_5 ={"happiness":0,"others":1,"disgust":2,"surprise":3,"repression":4}
SAMM_label_to_cls_5 ={"Happiness":0,"Other":1,"Anger":2,"Contempt":3,"Surprise":4}
# MMEW_label_to_cls_7 = {"happiness":0,"others":1,"disgust":2,"surprise":3,"fear":4,"anger":5,"sadness":6}
MMEW_label_to_cls_6 = {"happiness":0,"disgust":1,"surprise":2,"fear":3,"anger":4,"sadness":5}
SMIC_label_to_cls_3 ={"negative":0,"positive":1,"surprise":2}



class CASME2(data.Dataset):
    def __init__(self ,root, data_list_file ,test_subject, class_num = 7,sample = "interval",  phase="train", is_dynamic=False, input_shape=(3,224,224)):
        
        self.phase = phase
        self.input_shape = input_shape
        self.root = root
        self.sample = sample
        self.class_num = class_num
        self.is_dynamic = is_dynamic
        self.subject_name,self.file_name,self.onset_frame,self.offset_frame,\
            self.apex_frame,self.action_units,self.label = self.read_CASME2_xls(data_list_file,test_subject)
        
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomCrop(self.input_shape[1:],pad_if_needed=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.subject_name)

    def __getitem__(self,index):
        
        if self.apex_frame[index] == "/":
            self.apex_frame[index] = int((self.offset_frame[index]+self.onset_frame[index])/2)

        path = os.path.join(self.root,"sub"+str(self.subject_name[index]).zfill(2) +'/'+ str(self.file_name[index]))
        
        if self.class_num==7:
            label = CASME2_label_to_cls_7[self.label[index]]
        elif self.class_num==5:
            label = CASME2_label_to_cls_5[self.label[index]]

        if self.phase == "test":
            image_index_list = self.test_sample(index)
        else:
            if self.sample == "interval":
                image_index_list = self.interval_sample(index)
            elif self.sample == "apex":
                image_index_list = [self.onset_frame[index],self.apex_frame[index]]
            
        image_path_list = [os.path.join(path, "reg_img"+str(img)+'.jpg') for img in image_index_list]
        
        # if self.phase == "train":
        #     au_image_path_list = [os.path.join(path, "reg_img"+str(img)+'.jpg').replace("Cropped","Cropped_mask") for img in image_index_list]
        #     au_image_list = [self.transforms(Image.open(image).convert('RGB')).float() for image in au_image_path_list]

        # image_list = [os.path.join(path, "img"+str(img)+'.jpg') for img in image_list]

        # AU multi- label
        # .....

        if not self.is_dynamic:
            image_list = [self.transforms(Image.open(image).convert('RGB')).float() for image in image_path_list]
        else:
            image_list = self.generate_dynamic_image(image_list)
        
        apex_index = image_index_list.index(self.apex_frame[index])

        if self.phase == "train":
            return image_list,label,image_index_list,apex_index
        else:
            return image_list,label

    def generate_dynamic_image(self,image_list):
        # pdb.set_trace()
        dynamic_images = []
        images = [cv2.imread(image) for image in image_list]
        for i in range(3):
            dyn_image = get_dynamic_image(images[3*i:3*i+3], normalized=True)
            dyn_image = self.transforms(Image.fromarray(cv2.cvtColor(dyn_image,cv2.COLOR_BGR2RGB))).float()
            dynamic_images.append(dyn_image)

        return dynamic_images


    def interval_sample(self,index):
        if self.phase == "train":
            if self.offset_frame[index] - self.apex_frame[index] <=4 or self.apex_frame[index] - self.onset_frame[index] <=4:
                image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))
            else:
                image_index = list(np.random.randint(self.onset_frame[index],self.apex_frame[index]-1,(4,)))
                image_index += list(np.random.randint(self.apex_frame[index]+1,self.offset_frame[index],(4,)))
        else:
            image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))
            
        # image_index = []
        image_index.append(self.apex_frame[index])
        image_index = sorted(image_index)
        return image_index


    def test_sample(self,index):
        image_index = []
        for i in range(10):
            image_index.append(self.interval_sample(index))    
        image_index = sum(image_index,[])
        return image_index

    def read_CASME2_xls(self,path,test_subject):
        
        data = pd.read_excel(path)
        if self.class_num == 5:
            data = data[data["Estimated Emotion"]!="fear"]  
            data = data[data["Estimated Emotion"]!="sadness" ]
        if self.phase == "train":
            data = data[data["Subject"]!=test_subject]
        else:
            data = data[data["Subject"]==test_subject]

        subject_name = data["Subject"]
        file_name = data["Filename"]
        onset_frame = data["OnsetFrame"]
        offset_frame = data["OffsetFrame"]
        apex_frame = data["ApexFrame"]
        action_units = data["Action Units"]
        label = data["Estimated Emotion"]

        return subject_name.values.tolist(),file_name.values.tolist(),onset_frame.values.tolist(),\
        offset_frame.values.tolist(),apex_frame.values.tolist(),action_units.values.tolist(),label.values.tolist()


class SAMM(data.Dataset):

    def __init__(self ,root, data_list_file ,test_subject, class_num = 5,sample = "interval",  phase="train", is_dynamic=False, input_shape=(3,224,224)):
        
        self.phase = phase
        self.input_shape = input_shape
        self.root = root
        self.sample = sample
        self.class_num = class_num
        self.is_dynamic = is_dynamic
        self.subject_name,self.file_name,self.onset_frame,self.offset_frame,\
            self.apex_frame,self.action_units,self.label = self.read_SAMM_xls(data_list_file,test_subject)
        
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomCrop(self.input_shape[1:],pad_if_needed=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.subject_name)

    def __getitem__(self,index):

        path = os.path.join(self.root, str(self.subject_name[index]).zfill(3) +'/'+ str(self.file_name[index]))
        label = SAMM_label_to_cls_5[self.label[index]]

        if self.phase == "test":
            image_index_list = self.test_sample(index)
        else:
            if self.sample == "interval":
                image_index_list = self.interval_sample(index)
            elif self.sample == "apex":
                image_index_list = [self.onset_frame[index],self.apex_frame[index]]
            
        image_list = [os.path.join(path,  str(self.subject_name[index]).zfill(3)+'_'+str(img)+'.jpg') for img in image_index_list]
        image_list = [self.transforms(Image.open(image).convert('RGB')).float() for image in image_list]
        
        apex_index = image_index_list.index(self.apex_frame[index])

        if self.phase == "train":
            return image_list,label,image_index_list,apex_index
        else:
            return image_list,label
    
    def interval_sample(self,index):

        if self.offset_frame[index] - self.apex_frame[index] <=4 or self.apex_frame[index] - self.onset_frame[index] <=4:
            image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))
        else:
            image_index = list(np.random.randint(self.onset_frame[index],self.apex_frame[index]-1,(4,)))
            image_index += list(np.random.randint(self.apex_frame[index]+1,self.offset_frame[index],(4,)))

        image_index.append(self.apex_frame[index])
        image_index = sorted(image_index)
        
        return image_index


    def test_sample(self,index):
        image_index = []
        for i in range(10):
            image_index.append(self.interval_sample(index))    
        image_index = sum(image_index,[])
        return image_index

    def read_SAMM_xls(self,path,test_subject):

        data = pd.read_excel(path)
        if self.class_num == 5:
            data = data[data["Estimated Emotion"]!="Disgust"]  
            data = data[data["Estimated Emotion"]!="Fear" ]
            data = data[data["Estimated Emotion"]!="Sadness" ]
        if self.phase == "train":
            data = data[data["Subject"]!=test_subject]
        else:
            data = data[data["Subject"]==test_subject]

        subject_name = data["Subject"]
        file_name = data["Filename"]
        onset_frame = data["Onset Frame"]
        offset_frame = data["Offset Frame"]
        apex_frame = data["Apex Frame"]
        action_units = data["Action Units"]
        label = data["Estimated Emotion"]

        return subject_name.values.tolist(),file_name.values.tolist(),onset_frame.values.tolist(),\
        offset_frame.values.tolist(),apex_frame.values.tolist(),action_units.values.tolist(),label.values.tolist()
    
class SMIC(data.Dataset):

    def __init__(self ,root, data_list_file ,test_subject, class_num = 3,sample = "interval",  phase="train", is_dynamic=False, input_shape=(3,224,224)):

        self.phase = phase
        self.input_shape = input_shape
        self.root = root
        self.sample = sample
        self.class_num = class_num
        self.is_dynamic = is_dynamic
        self.subject_name,self.path,self.onset_frame,self.offset_frame,\
            self.apex_frame,self.label = self.read_SAMM_xls(data_list_file,test_subject)

        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomCrop(self.input_shape[1:],pad_if_needed=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.subject_name)

    def __getitem__(self,index):

        path = os.path.join(self.root, self.path[index])
        label = SMIC_label_to_cls_3[self.label[index]]

        if self.phase == "test":
            image_index_list = self.test_sample(index)
        else:
            if self.sample == "interval":
                image_index_list = self.interval_sample(index)
            elif self.sample == "apex":
                image_index_list = [self.onset_frame[index],self.apex_frame[index]]

        image_list = [os.path.join(path,  "reg_image"+str(img)+'.bmp') for img in image_index_list]
        image_list = [self.transforms(Image.open(image).convert('RGB')).float() for image in image_list]

        apex_index = image_index_list.index(self.apex_frame[index])

        if self.phase == "train":
            return image_list,label,image_index_list,apex_index
        else:
            return image_list,label

    def interval_sample(self,index):
        if self.phase == "train":
            if self.offset_frame[index] - self.apex_frame[index] <=4 or self.apex_frame[index] - self.onset_frame[index] <=4:
                image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))
            else:
                image_index = list(np.random.randint(self.onset_frame[index],self.apex_frame[index]-1,(4,)))
                image_index += list(np.random.randint(self.apex_frame[index]+1,self.offset_frame[index],(4,)))
        else:
            image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))

        image_index.append(self.apex_frame[index])
        image_index = sorted(image_index)

        return image_index

    def test_sample(self,index):
        image_index = []
        for i in range(10):
            image_index.append(self.interval_sample(index))
        image_index = sum(image_index,[])
        return image_index
    
    def read_SAMM_xls(self,path,test_subject):

        data = pd.read_excel(path)
        if self.phase == "train":
            data = data[data["subject_name"]!=test_subject]
        else:
            data = data[data["subject_name"]==test_subject]

        subject_name = data["subject_name"]
        path = data["path"]
        onset_frame = data["onset frame"]
        offset_frame = data["offset frame"]
        apex_frame = data["apex frame"]
        label = data["emotions"]

        return subject_name.values.tolist(),path.values.tolist(),onset_frame.values.tolist(),\
        offset_frame.values.tolist(),apex_frame.values.tolist(),label.values.tolist()


class MMEW(data.Dataset):

    def __init__(self ,root, data_list_file ,test_subject, class_num = 3,sample = "interval",  phase="train", is_dynamic=False, input_shape=(3,224,224)):
        
        self.phase = phase
        self.input_shape = input_shape
        self.root = root
        self.sample = sample
        self.class_num = class_num
        self.is_dynamic = is_dynamic
        self.subject_name,self.path,self.onset_frame,self.offset_frame,\
            self.apex_frame,self.label = self.read_MMEW_xls(data_list_file,test_subject)
        
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomCrop(self.input_shape[1:],pad_if_needed=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((256,256)),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.subject_name)

    def __getitem__(self,index):

        path = os.path.join(self.root, self.label[index]+'/'+self.path[index])
        label = MMEW_label_to_cls_6[self.label[index]]

        if self.phase == "test":
            image_index_list = self.test_sample(index)
        else:
            if self.sample == "interval":
                image_index_list = self.interval_sample(index)
            elif self.sample == "apex":
                image_index_list = [self.onset_frame[index],self.apex_frame[index]]
            
        image_list = [os.path.join(path,  str(img)+'.jpg') for img in image_index_list]
        image_list = [self.transforms(Image.open(image).convert('RGB')).float() for image in image_list]

        
        apex_index = image_index_list.index(self.apex_frame[index])

        if self.phase == "train":
            return image_list,label,image_index_list,apex_index
        else:
            return image_list,label
    
    def interval_sample(self,index):
        # if self.phase == "train":
        if self.offset_frame[index] - self.apex_frame[index] <=4 or self.apex_frame[index] - self.onset_frame[index] <=4:
            image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))
        else:
            image_index = list(np.random.randint(self.onset_frame[index],self.apex_frame[index]-1,(4,)))
            image_index += list(np.random.randint(self.apex_frame[index]+1,self.offset_frame[index],(4,)))
        # else:
        #     image_index = list(np.random.randint(self.onset_frame[index],self.offset_frame[index],(8,)))
            

        image_index.append(self.apex_frame[index])
        image_index = sorted(image_index)
        
        return image_index

    def test_sample(self,index):
        image_index = []
        for i in range(10):
            image_index.append(self.interval_sample(index))    
        image_index = sum(image_index,[])
        return image_index

    def read_MMEW_xls(self,path,test_subject):

        data = pd.read_excel(path)
        data = data[data["Estimated Emotion"]!="others"]
        if self.phase == "train":
            data = data[~data["Subject"].isin(test_subject)]
        else:
            data = data[data["Subject"].isin(test_subject)]

        subject_name = data["Subject"]
        path = data["Filename"]
        onset_frame = data["OnsetFrame"]
        offset_frame = data["OffsetFrame"]
        apex_frame = data["ApexFrame"]
        label = data["Estimated Emotion"]

        return subject_name.values.tolist(),path.values.tolist(),onset_frame.values.tolist(),\
        offset_frame.values.tolist(),apex_frame.values.tolist(),label.values.tolist()


# if __name__ == "__main__":
    
#     xls = "D:/LaoLingjie/Database/FER/mirco-Emotion/SAMM/SAMM_Micro_FACS_Codes_v2_my.xlsx"
    
#     data = pd.read_excel(xls)
#     subject_name = data["Subject"]
#     file_name = data["Filename"]
#     onset_frame = data["OnsetFrame"]
#     offset_frame = data["OffsetFrame"]
#     apex_frame = data["ApexFrame"]
#     action_units = data["Action Units"]
#     label = data["Estimated Emotion"]
