#Custom dataset class for GTA5

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda

class GTA5Custom(Dataset):
    def __init__(self, root_dir, height, width):
        super(GTA5Custom, self).__init__()
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.color_to_id = {
            (128, 64, 128):0,   #road
            (244, 35, 232): 1,  # sidewalk
            (70, 70, 70): 2,    # building
            (102, 102, 156): 3, # wall
            (190, 153, 153): 4, # fence
            (153, 153, 153): 5, # pole
            (250, 170, 30): 6,  # light
            (220, 220, 0): 7,   # sign
            (107, 142, 35): 8,  # vegetation
            (152, 251, 152): 9, # terrain
            (70, 130, 180): 10, # sky
            (220, 20, 60): 11,  # person
            (255, 0, 0): 12,    # rider
            (0, 0, 142): 13,    # car
            (0, 0, 70): 14,     # truck
            (0, 60, 100): 15,   # bus
            (0, 80, 100): 16,   # train
            (0, 0, 230): 17,    # motorcycle
            (119, 11, 32): 18   # bicycle
        }
        
        self.transform_image = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #mean and std from ImageNet
        ])

        self.transform_label = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64)))
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
        
        label = Image.open(self.labels[idx]).convert('RGB')
        label = self.transform_label(label)
        label = self.map_color_to_class(label)
        
        return image, label

    def __len__(self):
        return len(self.images)

    def map_color_to_class(self, label):
        # Convert the RGB image to a single integer value
        label_int = label[:, :, 0]*256*256 + label[:, :, 1]*256 + label[:, :, 2]

        # Create a tensor of the same shape as the input image, filled with 255 (the class id for 'unlabeled')
        class_id_image = torch.full_like(label_int, 255)

        # Replace each unique color value with the corresponding class id
        for color, class_id in self.color_to_id.items():
            color_int = color[0]*256*256 + color[1]*256 + color[2]
            class_id_image = torch.where(label_int == color_int, class_id, class_id_image)

        return class_id_image