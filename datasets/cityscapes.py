#Custom dataset class for Cityscapes

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda

class CityscapesCustom(Dataset):
    def __init__(self, root_dir, split, height, width):
        super(CityscapesCustom, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.height = height
        self.width = width
        
        self.transform_image = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #mean and std from ImageNet
        ])

        self.transform_label = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64)))
        ])

        self.images_dir = os.path.join(self.root_dir, 'images', self.split)
        self.labels_dir = os.path.join(self.root_dir, 'gtFine', self.split)

        self.cities_folders = os.listdir(self.images_dir)

        self.images = []
        self.labels = []

        for city in self.cities_folders:
            city_images = os.listdir(os.path.join(self.images_dir, city))
            city_labels = os.listdir(os.path.join(self.labels_dir, city))
            
            for city_image in city_images:
                self.images.append(os.path.join(self.images_dir, city, city_image))
                self.labels.append(os.path.join(self.labels_dir, city, city_image.replace('leftImg8bit', 'gtFine_labelTrainIds')))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('L')
        
        image = self.transform_image(image)
        label = self.transform_label(label)
        
        return image, label

    def __len__(self):
        return len(self.images)