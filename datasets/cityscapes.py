#Custom dataset class for Cityscapes

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CityscapesCustom(Dataset):
    def __init__(self, root_dir, split, transform=None):
        #super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

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
        #folder_name = self.images[idx].split('_')[0]
        #image_path = os.path.join(self.images_dir, folder_name, self.images[idx])
        #label_path = os.path.join(self.labels_dir, folder_name, self.images[idx].replace('leftImg8bit', 'gtFine_labelTrainIds'))

        #image = Image.open(image_path)#.convert('RGB')
        #image = np.array(Image.open(image_path.convert('RGB')))
        
        #label = Image.open(label_path)
        #label = np.array(Image.open(label_path).convert('L'), dtype=np.float32)

        #label[label == 255] = 19

        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        
        #if self.transform is not None:
            #image = self.transform[0](image)
            #label = self.transform[1](label)

        return image, label

    def __len__(self):
        return len(self.images)