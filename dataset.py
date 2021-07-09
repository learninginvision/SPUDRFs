import os.path
import torch
from glob import glob
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
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
        test_labels = []
        test_images_names = []
        test_weights = []

        with open(train_txt,"r") as f: 
            lines = f.readlines()   
            for line in lines:
                image_name, label, weight = line.strip('\n').split(' ')
                label = float(label)
                weight = float(weight)
                train_labels.append(label)
                train_images_names.append(image_name)
                train_weights.append(weight)
        with open(test_txt,"r") as f: 
            lines = f.readlines()      
            for line in lines:
                image_name, label, weight = line.strip('\n').split(' ')
                label = float(label)
                weight = float(weight)
                test_labels.append(label)
                test_images_names.append(image_name)
                test_weights.append(weight)
        if train:
            self.list = train_images_names
            self.labels = train_labels
            self.weights = train_weights
        else:
            self.list = test_images_names
            self.labels = test_labels
            self.weights = test_weights
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        filename = self.list[index]
        weight = self.weights[index]
        image = cv2.imread(os.path.join(self.image_dir, filename))[:, :, ::-1]
        image = image.astype(np.float32)
        label = np.float32(self.labels[index])
        return self.transform(image), label, weight

    def __len__(self):
        """Return the number of images."""
        return len(self.list)

def get_loader(train_txt='./MORPH-train.txt',\
     test_txt='./MORPH-test.txt',\
     image_dir='/root/data/meng/dataset/Morph_mtcnn_1.3_0.35_0.3/',\
     batch_size=32, image_size=224, num_workers=8, train=True, shuffle=True):

    transform = []

    if train:
        transform.append(T.Lambda(lambda x: random_flip(x, p=0.5)))
        transform.append(T.Lambda(lambda x: random_crop(x, (224, 224))))
    else:
        transform.append(T.Lambda(lambda x: center_crop(x, (224, 224))))
        
    transform.append(T.Lambda(lambda x: x-112.0))
    transform.append(T.Lambda(lambda x: torch.from_numpy(x.transpose((2,0,1))) ))
    transform = T.Compose(transform)

    dataset = DataMyload(train_txt, test_txt, image_dir, transform, train)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=train,
                                  num_workers=num_workers)
    return data_loader

def random_crop(img, output_size=(224, 224)):
    
    w, h, _ = img.shape
    tw, th = output_size
    i = random.randint(0, w - tw)
    j = random.randint(0, h - th)
    return crop(img, i, j, tw, th)

def center_crop(img, output_size=(224, 224)):
    
    w, h, _ = img.shape
    tw, th = output_size
    i = int(round((w - tw) / 2.))
    j = int(round((h - th) / 2.))
    return crop(img, i, j, tw, th)

def crop(img, i, j, tw, th):
    return img[i:tw+i, j:th+j, :]

def random_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, 1)
    else:
        return img


if __name__ == "__main__":
    train = get_loader(batch_size=5, train=True, shuffle=False)
    dataloader = iter(train)
    img, label, weight = next(dataloader)
    print(img.shape)
   