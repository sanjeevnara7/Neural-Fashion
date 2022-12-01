import os
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset
from PIL import Image 

#Define Custom Dataset Class
class FashionDataset(Dataset):
    def __init__(self, shape_file, fabric_file, pattern_file, root_dir, transform=None):
        super().__init__()
        self.shape_annotations = pd.read_csv(shape_file)
        self.fabric_annotations = pd.read_csv(fabric_file)
        self.pattern_annotations = pd.read_csv(pattern_file)

        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.shape_annotations.iloc[idx, 0]) # idx: row | 0: column(image name)
        image = Image.open(path)

        shape_tensor = torch.tensor(int(self.shape_annotations.iloc[idx, 1:]))
        fabric_tensor = torch.tensor(int(self.fabric_annotations.iloc[idx, 1:]))
        pattern_tensor = torch.tensor(int(self.pattern_annotations.iloc[idx, 1:]))

        y1 = torch.cat((shape_tensor, fabric_tensor, pattern_tensor))
        
        if self.transform: 
            image = self.transform(image)
        
        return image, y1