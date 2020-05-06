from __future__ import print_function, division

import torch
import torchvision.models as models
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import vgg11data
import time
import os
import copy


#
# https://www.kaggle.com/pytorch/vgg11bn#vgg11_bn.pth
# Downloaded above model to folder 'models' in ASLTranslator on local machine
#
# Portions of this code were adapted from a Pytorch tutorial on transfer learning,
# https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
#


num_to_letter = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q',
                 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
use_gpu = torch.cuda.is_available()

model = models.__dict__['vgg11_bn']()
model.load_state_dict(torch.load("models/vgg11_bn.pth"))

for param in model.features.parameters():
    param.require_grad = False


num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, len(num_to_letter))])  # Replace with output layer of correct size
model.classifier = nn.Sequential(*features)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

trainloader, train_size = vgg11data.create_data_set()
valloader, val_size = vgg11data.create_validation_set()

dataloaders = {
    'train': trainloader,
    'val': valloader
}
dataset_sizes = {
    'train': train_size,
    'val': val_size
}


def train_model(vgg, criterion1, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(dataloaders['train']):
            if i % 10 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches // 2), end='', flush=True)

            # Use half training dataset
            if i >= train_batches // 2:
                break

            inputs, labels = data

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion1(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataset_sizes['train']
        avg_acc = acc_train * 2 / dataset_sizes['train']

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion1(outputs, labels)

            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / dataset_sizes['val']
        avg_acc_val = acc_val / dataset_sizes['val']

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg


model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
torch.save(model.state_dict(), 'models/kaggle_vgg11.pt')
