import torch
import torch.nn as nn
import torch.nn.functional as F
from SignDataset import SignDataset
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import statanalysis

import cv2 as cv #need image processing because my picture sample size is wrong

import time
import copy

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3,6,5)

#pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

model_name = "resnet"
num_classes = 26
batch_size = 100
num_epochs = 5
#feature extracting changes how the model is fine-tuned, specifically if we are then only the last, output, layer is changed
feature_extract = True

#not sure if I need is_inception since it is for auxiliary output of architecture in sample, but doesn't hurt to keep for now
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()       #training mode
            else:
                model.eval()        #evaluate mode
            running_loss = 0.0
            running_corrects = 0

            for labels, inputs in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #zero the parameter gradients (not sure why this is necessary but I'll look into it later)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train': #pretty sure this is only for inception architecture which uses auxiliary
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + .4*loss2
                    else:
                        #print("inputs: {}".format(inputs))
                        #print(inputs.size())
                        outputs = model(inputs)
                        labels = labels.long()
                        loss = criterion(outputs,labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print('preds: {} '.format(preds))
                # print(labels.data)
                # temp = inputs[0].cpu().numpy()
                # print(temp)
                # plt.imshow(temp[0])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc. {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#model.fc = nn.Linear(512, num_classes)

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    #this is the only one im implementing right now, but i keep the format in case I want to try another one in the future
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
#print(model_ft)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229, 0,224, 0.225]),
    ])
}

#print("Initializing Datasets and Dataloaders...")

dataset = SignDataset(data_transforms[x] for x in ['train','val'])
dataloaders_dict = {x: DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) for x in ['train','val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum = 0.9)


criterion = nn.CrossEntropyLoss()


model_ft, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
statanalysis.plot(val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist)
torch.save(model_ft.state_dict(),'resnet_weight_5_epochs.pt')