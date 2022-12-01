import os
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset
from customDataset import FashionDataset
import shutil
import pandas as pd

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Image Augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize()
])

# Load Data


shutil.unpack_archive('labels.zip')

shape_file = './labels/shape/shape_anno_all.txt'
fabric_file = './labels/texture/fabric_ann.txt'
pattern_file = './labels/texture/pattern_ann.txt'

dataset = FashionDataset(shape_file, fabric_file, pattern_file, 'images', transform)

train, test = torch.utils.data.random_split(dataset, [40000, 4000])

train_loader = DataLoader(dataset = train, batch_size = 64, shuffle = True)
test_loader = DataLoader(dataset = test, batch_size = 64, shuffle = True)










