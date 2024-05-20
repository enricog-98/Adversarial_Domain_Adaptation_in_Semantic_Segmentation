#Custom dataset class for GTA5

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda

#Need to be modified

class GTA5Custom(Dataset):
    def __init__(self, root_dir, height, width):
        super(GTA5Custom, self).__init__()
        self.root_dir = root_dir
        self.height = height
        self.width = width

        #Mapping of ignore categories (255) and valid ones (in range 0-18)
        self.mapping_19 = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
                           10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
                           20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
                           30: 255, 31: 16, 32: 17, 33: 18, -1: 255
        }
        
        self.transform_image = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #mean and std from ImageNet
        ])

        self.transform_label = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64))),
            #transforms.ToTensor(),
        ])

        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')

        self.images = []
        self.labels = []

        for image in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, image))
            self.labels.append(os.path.join(self.labels_dir, image))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform_image(image)
        
        label = Image.open(self.labels[idx])
        label = self.transform_label(label)
        label = self.encode_labels(label)
        
        return image, label

    def __len__(self):
        return len(self.images)
    
    def encode_labels(self, mask):
        label_mask = np.zeros_like(mask)
        for k in self.mapping_19:
            label_mask[mask == k] = self.mapping_19[k]
        return label_mask