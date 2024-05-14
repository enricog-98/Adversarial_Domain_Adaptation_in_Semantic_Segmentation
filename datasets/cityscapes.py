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

        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        #self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        #self.class_names = ["unlabelled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
        #                    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck",
        #                    "bus", "train", "motorcycle", "bicycle"
        #]

        #self.ignore_index = 255
        #self.class_map = dict(zip(self.valid_classes, range(19)))

        # Mapping of ignore categories and valid ones (numbered from 1-19)
        #self.mapping_20 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0,
        #                   10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0, 19: 7,
        #                   20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 0,
        #                   30: 0, 31: 17, 32: 18, 33: 19, -1: 0
        #}

        #Mapping of ignore categories (255) and valid ones (numbered from 0-18)
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
        image = self.transform_image(image)
        
        label = Image.open(self.labels[idx])
        label = self.transform_label(label)
        label = self.encode_labels(label)#0 to 19
        #label = self.encode_segmap(label)#0 to 255
        
        return image, label

    def __len__(self):
        return len(self.images)
    
    #def encode_segmap(self, mask):
        #for _voidc in self.void_classes:
            #mask[mask == _voidc] = self.ignore_index
        #for _validc in self.valid_classes:
            #mask[mask == _validc] = self.class_map[_validc]
        #return mask
    
    def encode_labels(self, mask):
        label_mask = np.zeros_like(mask)
        for k in self.mapping_19:
            label_mask[mask == k] = self.mapping_19[k]
        return label_mask