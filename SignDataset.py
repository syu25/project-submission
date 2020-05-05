import os
import cv2
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
import torch.nn.functional as F

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


class SignDataset(Dataset):

    #def __init__(self, csv_file, root_dir, transform=None):
    def __init__(self, transform = None):

        #print('initializing dataset')

        self.labels = np.loadtxt('.\\data\\training_labels\\labels.csv')

        self.image_names = os.listdir('.\\data\\training_images\\combined')
        self.image_names.sort()
        self.image_names = [os.path.join('.\\data\\training_images\\combined', img_name) for img_name in self.image_names]

        #print(self.image_names)
        #print(self.labels)

        self.transform = transform

        #self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        letter = self.labels[idx]
        #print(self.image_names[idx])
        #image = Image.open(self.image_names[idx])
        image = io.imread(self.image_names[idx])

        if self.transform:
            #image = self.transform(image)

            image = cv2.resize(image,dsize=(224,224))
            #No idea how this enumerate is supposed to work so i'm just hardcoding the transformations for now
            #for i, f in enumerate(self.transform):
            #    newimage = f(image)
            #
            transform = transforms.ToTensor()
            image = transform(image)
            transform = transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
            image = transform(image)
        return letter, image


#            if(torch.is_tensor(idx)):
#                idx = idx.tolist()
#
#                img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])
#                image = io.imread(img_name)
#                landmarks = self.landmarks_frame.iloc[idx,0]

# dataset = SignDataset()
# images = []
# for n in dataset.image_names:
#     print("filepath: " + n)
#     images.append(np.array(Image.open(n)))
# a, b = dataset[0]
#
# print("a: {}".format(a))
# b.show()