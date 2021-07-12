import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import random
import os
import numpy as np
T = transforms

class DataMyload(Dataset):
    '''
    :param
    train_txt: train.txt; 
    test_txt: test.txt; 
    image_dir: image path; 
    transform: transform; 
    train: if Ture load train.txt, else load test.txt
    '''
    def __init__(self, train_txt, test_txt, image_dir, transform, train=True):
        train_labels = []
        train_images_names = []
        train_weights = []
        train_headpose = []
        test_labels = []
        test_images_names = []
        test_weights = []
        test_headpose = []

        with open(train_txt,"r") as f: 
            lines = f.readlines()   
            for line in lines:
                info = line.strip('\n').split(' ')
                lname, rname, pitch, yaw, weight = info[0], info[1], info[2], info[3], info[4]
                pitch = float(pitch)
                yaw = float(yaw)
                weight = float(weight)
                headpose = info[5:]
                train_labels.append([pitch, yaw])
                train_images_names.append([lname, rname])
                train_weights.append(weight)
                train_headpose.append(headpose)

        with open(test_txt,"r") as f: 
            lines = f.readlines()      
            for line in lines:
                info = line.strip('\n').split(' ')
                lname, rname, pitch, yaw, weight = info[0], info[1], info[2], info[3], info[4]
                pitch = float(pitch)
                yaw = float(yaw)
                weight = float(weight)
                headpose = info[5:]
                test_labels.append([pitch, yaw])
                test_images_names.append([lname, rname])
                test_weights.append(weight)
                test_headpose.append(headpose)
        if train:
            self.list = train_images_names
            self.labels = train_labels
            self.weights = train_weights
            self.headpose = train_headpose
        else:
            self.list = test_images_names
            self.labels = test_labels
            self.weights = test_weights
            self.headpose = test_headpose
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        filename = self.list[index]
        lname, rname = filename[0], filename[1]
        weight = self.weights[index]
        
        lefteye = Image.open(os.path.join(self.image_dir, lname)).convert("RGB")
        righteye = Image.open(os.path.join(self.image_dir, rname)).convert("RGB")


        label = np.float32(self.labels[index]) /1.0
        headpose = np.float32(self.headpose[index])
        return self.transform(lefteye), self.transform(righteye), label, weight, headpose

    def __len__(self):
        """Return the number of images."""
        return len(self.list)

def get_loader(train_txt= './txt/2train-FGNET.txt',\
     test_txt='./txt/2test-FGNET.txt',\
     image_dir='/root/data/meng/dataset/FGNET_normalized/',\
     batch_size=32, image_size=224, num_workers=2, train=True, shuffle=True):

    transform = []

    if train:
        transform.append(T.Lambda(lambda x: randTransform3(x)))
        transform.append(T.ToTensor())
    else:

        transform.append(T.ToTensor())
        
    transform.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform = T.Compose(transform)

    dataset = DataMyload(train_txt, test_txt, image_dir, transform, train)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=train,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader

def randTransform(img):
    seed = random.uniform(0, 1)
    base = 1.0/6.0
    if seed < base:
        aug = T.Resize((39, 65))(img)
        aug = T.RandomCrop((36, 60))(aug)
    elif seed < 2.0*base:
        aug = T.Grayscale(num_output_channels=3)(img)
    elif seed < 3.0*base:
        aug = ImageOps.equalize(img)
    elif seed < 4.0*base:
        aug = T.Resize((18, 30), interpolation=0)(img)
        aug = T.Resize((36, 60), interpolation=2)(aug)
    elif seed < 5.0*base:
        aug = T.Resize((9, 15), interpolation=0)(img)
        aug = T.Resize((36, 60), interpolation=2)(aug)
    elif seed <= 6.0*base:
        aug = img
    return aug

def randTransform1(img):
    seed = random.uniform(0, 1)
    base = 1.0/3.0
    if seed < base:
        aug = T.Resize((39, 65))(img)
        aug = T.RandomCrop((36, 60))(aug)
    elif seed < 2.0*base:
        aug = T.Grayscale(num_output_channels=3)(img)
        aug = ImageOps.equalize(aug)
    elif seed <= 3.0*base:
        aug = img
    return aug

def randTransform2(img):
    aug = T.Grayscale(num_output_channels=3)(img)
    aug = ImageOps.equalize(aug)
    return aug

def randTransform3(img):

    seed = random.uniform(0, 1)
    base = 1.0/2.0
    aug = img
    if seed < base:
        aug = T.Resize((39, 65))(img)
        aug = T.RandomCrop((36, 60))(aug)
    
    return aug
