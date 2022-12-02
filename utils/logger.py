from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import os

class DataLogger():
    def _init_(self, model_name = 'Model'):  
        #create the logging directory based on the model_name
        self.log_dir = os.path.join("saved_models", model_name)
        os.makedirs(self.log_dir, exist_ok = True)

        #initialize the SummaryWriter with the logging directory path
        self.writer = SummaryWriter(self.log_dir)
        return
    
    def log_loss_accuracy(self, loss, acc, mode, epoch):
        #add the loss and accuracy values of each train/validation step
        self.writer.add_scalar(mode+" loss", loss, epoch)
        self.writer.add_scalar(mode+" accuracy", acc, epoch)
        return
    
    def log_images(self, image_data, index, mode = 'train'):
        #transform to un-normalize the image for visualization
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                               ])

        unnorm_image_data = invTrans(image_data)
        #display 6 images per row on tensorboard
        grid = torchvision.utils.make_grid(unnorm_image_data, nrow = 6)
        self.writer.add_image(f'{mode}_images', grid, index)

    def visualize_logged_data(self):
        #visualize the logged data on tensorboard
        %load_ext tensorboard
        %tensorboard --logdir={"saved_models"}
        
        return
