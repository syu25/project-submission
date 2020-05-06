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

class LocalizationDataset(Dataset):
    def __init__(self, i):

        self.image = i

        self.transform = transform
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = self.image

        if self.transform:
            #image = cv2.resize(image,dsize=(224,224))
            transform = transforms.ToTensor()
            image = transform(image)
        return image, None