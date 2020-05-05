import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
import io
import os
import json
from PIL import Image

model = torch.load('model.pth', map_location=torch.device('cpu'))
def main():
    model.eval()
    data_dir = os.path.dirname(os.path.abspath(__file__))
    with open(data_dir+'/'+'fish.png', 'rb') as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes=image_bytes)
    print(tensor)
    temp=(get_prediction(image_bytes))
    print(temp)
    fishy=fishFunction(temp)
    #print("Prediction: " + str(get_prediction(image_bytes)))
    return "Prediction: "+fishy



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx

def fishFunction(input):
    if input=="0":
        name="Two-stripe Damselfish" #Dascyllus reticulatus
    elif input=="1":
        name="Blackbar Devil" #Plectroglyphidodon dickii
    elif input=="2":
        name="Yellowtail Reeffish"  #Chromis chrysura
    elif input=="3":
        name="Yellowtail Clownfish "#Amphiprion clarkii
    elif input=="4":
        name="Oval butterflyfish" #Chaetodon lunulatus
    elif input=="5":
        name="Chevron butterflyfish" #Chaetodon trifascialis
    elif input=="6":
        name="Myripristis kuntee" #Myripristis kuntee
    elif input=="7":
        name="Spot-cheeked surgeonfish" #Acanthurus nigrofuscus
    elif input=="8":
        name="Barred thicklip" #Hemigymnus fasciatus
    elif input=="9":
        name="Sammara squirrelfish" #Neoniphon sammara
    elif input=="10":
        name="Sergeant major" #Abudefduf vaigiensis
    elif input=="11":
        name="Canthigaster valentini" #Canthigaster valentini
    elif input=="12":
        name="Lemon damselfish" #Pomacentrus moluccensis
    elif input=="13":
        name="Brown tang" #Zebrasoma scopas
    elif input=="14":
        name="Blackeye thicklip wrasse" #Hemigymnus melapterus
    elif input=="15":
        name="Blacktail snapper" #Lutjanus fulvus
    else:
        name="not recognized"

    return name

if __name__ == '__main__':
    main()
