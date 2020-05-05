import os
import numpy as np
import torch
from PIL import Image
import scipy.io
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import csv


class EgohandsDataset(Dataset):
    def __init__(self, transforms):
        #self.root = root
        self.transforms = transforms
        self.root = './egohands_processed'
        size = len(os.listdir('./egohands_processed/images'))
        self.imgs = []
        self.masks = []
        self.annotations = []
        for i in range(0,size):
            self.imgs.append('image{}.png'.format(i))
            self.masks.append('mask{}.png'.format(i))
            self.annotations.append('annotation{}.txt'.format(i))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        mask_path = os.path.join(self.root, 'masks', self.masks[idx])
        annotation_path = os.path.join(self.root, 'annotations', self.annotations[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        #print(mask_path)
        mask = np.array(mask)
        #obj_ids = np.unique(mask)
        obj_ids = np.unique(mask.reshape(-1,mask.shape[2]),axis=0)
        obj_ids = obj_ids[1:]
        #print(obj_ids.shape)
        masks = (mask == obj_ids[:, None, None])[:,:,:,0]
        #this is 3 channels for some reason when it should be 1 channel so NxHxWxC -> NxHxW

        #print(masks.shape)
        #print(masks[0][570][500])
        #print(obj_ids)
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            #pos = np.where(np.all(mask == obj_ids[i]))
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        #print(boxes)
        #print('area: {} - {} * {} - {}'.format(boxes[:, 3], boxes[:, 1], boxes[:, 2], boxes[:, 0]))
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


        # cv2.rectangle(mask, (xmin, ymin), (xmax,ymax), (0,255,0),1)
        # cv2.imshow('test',mask)
        # cv2.waitKey(0)