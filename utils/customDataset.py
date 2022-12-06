import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# Code to define custom dataset
from PIL import Image 
import pandas as pd
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import tensorflow as tf

#Define Custom Dataset Class
class FashionDataset(Dataset):
    def __init__(self, data_np, captions, tokenizer, root_dir, mode='train'):
        super().__init__()
        self.data_np = data_np
        self.root_dir = root_dir
        self.mode = mode
        self.captions = captions
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        filename = self.data_np[idx, 0]
        path = os.path.join(self.root_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image from BGR to RGB format
        y1 = torch.from_numpy(self.data_np[idx, 1:].astype('float32'))
        apply_transform = self.transform_data()
        image = apply_transform(image = img)['image']
        
        y2 = self.captions.loc[filename]
        y2 = torch.tensor(y2['sequence'])
        return image, y1, y2
    
    #Function to apply transforms
    def transform_data(self):
        #Augmentations during training
        if self.mode == 'train':
            transform_data = A.Compose(
              [
                  A.HorizontalFlip(p=0.4),  
                  A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0, rotate_limit=15, p=0.5),
                  #randomly change brightness, contrast, and saturation of the image 50% of the time
                  A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue = 0, p=0.5), 
                  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1), 
                  ToTensorV2(p=1),
              ])
        else:     
          #augmentations during validation and testing
          transform_data = A.Compose(
          [
              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1),
              ToTensorV2(p=1),
          ])
    
        return transform_data