import os
import sys

import numpy 
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import conf
from utils import image_preprocess

def get_imagefolder_train_loader():
    train_dir = './data/ImageFolder/train'

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(conf.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = conf.TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = conf.NUM_WORKERS,
        pin_memory = True
    )

    return train_loader

def get_imagefoler_val_loader():
    val_dir = './data/ImageFolder/test'

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
            transforms.Resize(int(conf.IMAGE_SIZE / 0.875)),
            transforms.CenterCrop(conf.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = conf.VAL_BATCH_SIZE,
        shuffle = False,
        num_workers = conf.NUM_WORKERS,
        pin_memory = True
    )

    return val_loader

class MyDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height = 256, resize_width = 256, repeat = 1):
        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.toTensor = transforms.ToTensor()
 
    def __getitem__(self, i):
        index = i % self.len
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization = False)
        img = self.data_preproccess(img)
        label = numpy.array(label)
        return img, label.squeeze()
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
 
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list
 
    def load_data(self, path, resize_height, resize_width, normalization):
        image = image_preprocess(path, resize_height, resize_width, normalization)
        return image
 
    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data

def get_custom_train_loader():
    data_dir = './data/Custom/train/images/'
    txt_dir = './data/Custom/train/metadata.txt'

    my_dataset = MyDataset(txt_dir, data_dir)

    train_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size = conf.TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = conf.NUM_WORKERS
    )

    return train_loader

def get_custom_val_loader():
    data_dir = './data/Custom/test/images/'
    txt_dir = './data/Custom/test/metadata.txt'

    my_dataset = MyDataset(txt_dir, data_dir)

    val_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size = conf.VAL_BATCH_SIZE,
        shuffle = False,
        num_workers = conf.NUM_WORKERS
    )

    return val_loader
