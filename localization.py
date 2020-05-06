from EgohandsDataset import EgohandsDataset
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from references.detection.engine import train_one_epoch, evaluate
import math
import sys
import references.detection.transforms as T
import references.detection.utils

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




# dataset = EgohandsDataset()
# img, target = dataset[0]
# plt.imshow(img)
# plt.show()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
dataset = EgohandsDataset(get_transform(train=True))
dataset_test = EgohandsDataset(get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()
#dataset = torch.utils.data.Subset(dataset, indices[0:19])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[20:30])
dataset = torch.utils.data.Subset(dataset, indices[:-788])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-787:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=references.detection.utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=references.detection.utils.collate_fn)

model = get_model_instance_segmentation(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    lr_scheduler.step()
    #img, target = data_loader_test.dataset[2]
    #print(target['image_id'])
    #print(len(target['masks']))
    evaluate(model, data_loader_test, device=device)
    print('evaluated')
    torch.save(model.state_dict(), 'rcnn_weight_1_epochs.pt')
print("finish")