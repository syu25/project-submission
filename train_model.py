import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from SignDataset import SignDataset


#class train_model():

fig=plt.figure(figsize=(8,8))
rows = 3
columns = 3

dataset = SignDataset()
for i in range(len(dataset)):
    letter,image = dataset[i]

    fig.add_subplot(rows,columns,i+1)

    print(letter)
    plt.imshow(image)

    if i == 8:
        plt.show()
        break

plt.clf()
fig=plt.figure(figsize=(8,8))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
dataiter = iter(dataloader)
letter, image = dataiter.next()
print(letter)
counter = 1
for i in image:
    fig.add_subplot(3,4, counter)
    plt.imshow(i)
    counter+=1
plt.show()