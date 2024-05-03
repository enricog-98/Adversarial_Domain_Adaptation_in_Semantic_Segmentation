#Custom dataset class for Cityscapes

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class CityscapesCustom(Dataset):
    def __init__(self, root_dir, split):
        super(CityscapesCustom, self).__init__()
        self.root_dir = root_dir
        self.split = split
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        image = self.transform(image)
        label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.images)