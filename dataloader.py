import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
#from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob

def fft_resove(img):
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift[dft_shift==0] += 0.1
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow,ccol = int(rows/2) , int(cols/2)
    
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    img_back = cv2.dft(np.float32(img_back), flags=cv2.DFT_COMPLEX_OUTPUT)
    low_dft_shift = np.fft.fftshift(img_back)
    low_dft_shift[low_dft_shift==0] += 0.1
    #if 0 in low_dft_shift:

    low_magnitude_spectrum = 20 * np.log(cv2.magnitude(low_dft_shift[:, :, 0], low_dft_shift[:, :, 1]))
    
    
    mask2 = np.ones((rows,cols,2),np.uint8)
    mask2[crow-30:crow+30, ccol-30:ccol+30] = 0
    fshift = dft_shift*mask2
    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    dft1 = cv2.dft(np.float32(img_back), flags=cv2.DFT_COMPLEX_OUTPUT)
    high_dft_shift = np.fft.fftshift(dft1)
    high_dft_shift[high_dft_shift==0] += 0.1


    high_magnitude_spectrum1 = 20 * np.log(cv2.magnitude(high_dft_shift[:, :, 0], high_dft_shift[:, :, 1]))

    return magnitude_spectrum, low_magnitude_spectrum, high_magnitude_spectrum1


class Dataload():
    def __init__(self, file_path, batch_size = 1, data_source = None, gray = False, image_shape = (128,128), dataset_type = "train", keep_same = True, limit = None, need_fft = False,shuffle = True):
        self.file_path = file_path
        self.data_source = data_source
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dataset_type = dataset_type
        self.limit = limit
        self.fft = need_fft
        self.datagen_fft = None
        if(gray):
            self.channels = 1
        else:
            self.channels = 3
        self.load_data(file_path, keep_same, shuffle = shuffle)
        self.set_gan()
        
        # self.X_train, self.Y_train = self.load_all_data(False ,gray, "train")
        # self.X_val, self.Y_val = self.load_all_data(False ,gray, "val")
        
    def check_dir(self, path):
        if (not os.path.exists(path)):
            return 0
        return 1
    
    def read_image_data(self, file_path, gray = False):
        if(gray):
            image = cv2.imread(file_path, 0) 
        else:
            image = cv2.imread(file_path)
        if(image is None):
             raise RuntimeError('image can \'t read:' + file_path)
        return image

    def set_gan(self):
        cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
        cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
        if(self.dataset_type == "train"):
            if self.channels == 1:
                self.datagen = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize(self.image_shape),
                #transforms.RandomCrop(self.image_shape[0], padding= int(self.image_shape[0]/8)),
                #transforms.RandomRotation(5),  # 随机旋转
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar_norm_mean, cifar_norm_std),
                transforms.Grayscale(1),
            ])
            else:
                self.datagen = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize(self.image_shape),
                #transforms.RandomCrop(self.image_shape[0], padding= int(self.image_shape[0]/8)),
                transforms.ColorJitter(0.2, 0.2, 0.2),  # 随机颜色变换,
                #transforms.RandomRotation(5),  # 随机旋转
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar_norm_mean, cifar_norm_std),
            ])
        else:
            if self.channels == 1:
                self.datagen = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.image_shape),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
                    transforms.Grayscale(1),
                ])
            else:
                self.datagen = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.image_shape),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar_norm_mean, cifar_norm_std),

                ])
        self.datagen_fft = transforms.Compose([
                transforms.ToTensor(),
        ])

    def load_data(self, file_path , keep_same = True, shuffle = True):
        self.file_path_list = []
        self.label_list = []
        self.label_list_True = []
        self.label_list_False = []
        self.lable_list_val = []
        self.label_dict = {}
        if (self.dataset_type == "train"):
            middle = file_path + "/trainset/"
            label_path = file_path + "/trainset_label.txt"
        elif(self.dataset_type == "val"):

            file = os.listdir(file_path)
            if(len(file) != 2):
                raise RuntimeError("file not val type file")
            for file_object in file:
                if(file_object[-4:] == ".txt"):
                    label_path =file_path + "/" + file_object 
                else:
                    middle = file_path + "/" + file_object + "/"
            #label_path = file_path + "/trainset_label.txt"
        
        else:
            middle =  file_path + "/valset/"
            label_path = file_path + "/valset_label.txt"
        print(middle, label_path)
        if(not self.check_dir(middle)):
            if(self.dataset_type == "train"):
                raise RuntimeError('train dir not exists:'+ middle)
            else:
                raise RuntimeError('val dir not exists:' + middle)
            
        file_path_list = glob.glob(middle+"*.jpg")
        with open(label_path,"r") as f:
            for line in f.readlines():
                d = line.split(" ")
                if(self.dataset_type == "val"):
                    label_name = d[0] 
                    label_gt = -1
                else:
                    label_name = d[0]
                    label_gt = int( d[1].replace("\n","") )
                self.label_dict[label_name] = label_gt

               
        for i in file_path_list:
            _, name = i.split("\\")
            gt = self.label_dict[name[:-4]]
            self.file_path_list.append([i, gt])
            if(gt == 1):
                self.label_list_True.append([i, gt])
            elif(gt == 0):
                self.label_list_False.append([i, gt])
            else:
                self.lable_list_val.append([i,gt])
        if(shuffle):
            random.shuffle(self.label_list_False)
            random.shuffle(self.label_list_True)
        if(keep_same):
            if self.limit is None:
                end = min(len(self.label_list_True), len(self.label_list_False))
            else:
                end = self.limit
                
            self.label_list_False = self.label_list_False[:end]
            self.label_list_True = self.label_list_True[:end]

        print("total:", len(self.file_path_list))
        print("True {}, False {}, val {}".format( len(self.label_list_True), len(self.label_list_False), len(self.lable_list_val)))

        self.len_true = len(self.label_list_True)
        self.len_false =  len(self.label_list_False)
        self.len_val = len(self.lable_list_val)



    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        try:
            a = index 
            if(self.len_true != 0 and self.len_false != 0):
                if(index >= self.len_true):
                    index = index - self.len_true
                    image = self.read_image_data( self.label_list_False[index][0] )
                    label = float(self.label_list_False[index][1])
                else:
                    image = self.read_image_data( self.label_list_True[index][0] )
                    label = float(self.label_list_True[index][1])
            else:
                image = self.read_image_data( self.lable_list_val[index][0] )
                label = self.lable_list_val[index][0]  # path

            w = image.shape[0]
            h = image.shape[1]
            if(w,h) != self.image_shape:
                image = cv2.resize(image, self.image_shape)

            if self.fft :
                if(len(image.shape) == 3):
                    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image2 = image

                origin,low,high = fft_resove(image2)
                if self.datagen_fft is not None:
                    origin = self.datagen_fft(origin)
                    low = self.datagen_fft(low)
                    high = self.datagen_fft(high)
            
            if self.datagen is not None:
                image = self.datagen(image)
            

        except Exception as e:
            print(e)
            print(a, index, self.len_false, self.len_true)
            raise ValueError("out of index")
        if self.dataset_type == "val":
            if self.fft:
                return image, label, origin, low, high
            else:
                return image, label
        else:

            if self.fft:
                return image, torch.tensor(label), origin, low, high
            else:
                return image, torch.tensor(label)


    def __len__(self):
        return self.len_false + self.len_true + self.len_val
    
         

if __name__ == '__main__':
    
    batch_size = 32
    
    
    train_dataloader = Dataload("H:/DATASET/phase1", dataset_type = "train")
    test_dataloader = Dataload("H:/DATASET/phase1", dataset_type = "test")
    train_loader= DataLoader(
        dataset = train_dataloader,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
    )
    test_loader= DataLoader(
        dataset = test_dataloader,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
    )