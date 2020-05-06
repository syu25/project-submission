import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torchvision import datasets, models
from torch.autograd import Variable
import copy
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader
from TestDataset import TestDataset
import matplotlib.pyplot as plt


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from LocalizationDataset import LocalizationDataset

if __name__ == '__main__':
    print("Hello, World")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX



    signs = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    #First parameter is videocapture property, accessed by number
    cap.set(3, 1280) #set Width = 300
    cap.set(4, 720) #set Height = 300
    #cap.set(12, 0.1)

    bounded_box_height = cap.get(4)
    bounded_box_width = cap.get(3)
    dimension = 224

    #resnext = torchvision.models.resnext101_32x8d(pretrained=True,progress=True)


    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 26)
    input_size = 224

    model.load_state_dict(torch.load('resnet_weight.pt'))
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #Localization
    local_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
    in_features = local_model.roi_heads.box_predictor.cls_score.in_features
    local_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = local_model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    local_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    local_model.load_state_dict(torch.load('rcnn_weight_10_epochs.pt'))
    local_model.eval()

    last = time.time()
    lastbox = time.time()
    #This displays on the screen what the predicted sign is
    sign = 'DEFAULT'
    left_bounds = []
    right_bounds = []
    top_bounds = []
    bottom_bounds = []

    capzone_top = 240
    capzone_bot = 480
    capzone_left = 460
    capzone_right = 820

    while(True):
        ret, original = cap.read()

        frame = copy.deepcopy(original)
        left_bound = int(bounded_box_width / 2 - dimension / 2)
        right_bound = left_bound + dimension
        top_bound = int(bounded_box_height / 2 - dimension / 2)
        bottom_bound = top_bound + dimension
        croppedFrame = frame[top_bound:bottom_bound, left_bound:right_bound]  # x and y are flipped idk why

        if time.time()-lastbox > 60:
            lastbox = time.time()
            with torch.no_grad():
                dataset_test = LocalizationDataset(Image.fromarray(frame[capzone_top:capzone_bot, capzone_left:capzone_right]))
                img, target = dataset_test[0]
                prediction = local_model([img])
                left_bounds.clear()
                right_bounds.clear()
                top_bounds.clear()
                bottom_bounds.clear()
                masks = prediction[0]['masks']
                #Image.fromarray(prediction[0]['masks'][0,0].mul(255).byte().cpu().numpy()).show()
                for boxes in prediction[0]['boxes']:
                    if (boxes[2] - boxes[1]) > 5 and (boxes[3] - boxes[1]) > 5:
                        left_bounds.append(boxes[0])
                        right_bounds.append(boxes[2])
                        top_bounds.append(boxes[1])
                        bottom_bounds.append(boxes[3])
                    #print(boxes)
                    #frame = cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0,255,0), thickness=2, lineType=8, shift=0)
        #print('{} to {} out of {}'.format(left_bound, right_bound, bounded_box_width))
        #print('{} to {} out of {}'.format(top_bound, bottom_bound, bounded_box_height))

        #frame = cv2.rectangle(frame,(left_bound,top_bound),(right_bound,bottom_bound),0,5,8,0)
        frame = cv2.rectangle(frame, (capzone_left,capzone_top),(capzone_right,capzone_bot),(255,255,255),2,8,0)
        #print('frame: {}'.format(frame.shape))
        #print(len(left_bounds))
        avg_box_left = 0
        avg_box_right = 0
        avg_box_top = 0
        avg_box_bot = 0
        for boxes in range(len(left_bounds)):
            box_left = int(left_bounds[boxes]) + capzone_left
            box_right = int(right_bounds[boxes]) + capzone_left #All shifted rigth by the amount of left bound
            box_top = int(top_bounds[boxes]) + capzone_top
            box_bot = int(bottom_bounds[boxes]) + capzone_top #All shifted down by the amount of top bound
            avg_box_left = avg_box_left + box_left
            avg_box_right = avg_box_right + box_right
            avg_box_top = avg_box_top + box_top
            avg_box_bot = avg_box_bot + box_bot
            frame = cv2.rectangle(frame,
                                  (box_left, box_top),
                                  #(capzone_left, capzone_top),
                                  (box_right, box_bot),
                                  #(capzone_right, capzone_bot),
                                  (0, 255, 0), thickness=2,
                                  lineType=8, shift=0)
            #print('[{}:{}, {}:{}]'.format((box_top-100),(box_bot+100),(box_left-100),(box_right+100)))
            #100's are just a general margin of error cuz the box fitting is pretty tight
            #croppedFrame = frame[(box_top-100):(box_bot+100), (box_left-100):(box_right+100)] #x-y flipped
        if len(left_bounds) > 0:
            avg_box_left = avg_box_left / len(left_bounds)
            avg_box_right = avg_box_right / len(left_bounds)
            avg_box_top = avg_box_top / len(left_bounds)
            avg_box_bot = avg_box_bot / len(left_bounds)
            left = int((avg_box_right+avg_box_left) / 2 - dimension / 2)
            right = left + dimension
            top = int((avg_box_top+avg_box_bot) / 2 - dimension / 2)
            bottom = top + dimension
            croppedFrame = original[top:bottom, left:right]  # x and y are flipped idk why
            print('{}:{}, {}:{}'.format(top, bottom, left, right))
            frame = cv2.rectangle(frame, (left,top), (right, bottom), (255,0,0), thickness=2, lineType=8, shift=0)
            #croppedFrame = frame[(box_top - 100):(box_bot + 100), (box_left - 100):(box_right + 100)]  # x-y flipped
        else:
            frame = cv2.rectangle(frame, (left_bound,top_bound), (right_bound, bottom_bound), (255,0,0), thickness=2, lineType=8, shift=0)
            croppedFrame = original[top_bound:bottom_bound, left_bound:right_bound]  # x-y flipped

        print(croppedFrame.shape)
        #Gets rid of the info on the bottom bar
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame = cv2.putText(frame, sign, (10,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF == ord(' ')): #27= esc key but I don't know where the constant is being kept  #ord(' ')):
            out = cv2.imwrite('capture.jpg',frame)
            break

        if time.time()-last > 1:
            last = time.time()
            input = croppedFrame
            dataloader = DataLoader(TestDataset(input), batch_size=1, shuffle=False, num_workers=0)
            for i in dataloader:
                outputs = model(i)
                _, preds = torch.max(outputs, 1)
                #print(outputs)
            sign = signs[preds.int()].upper()

    cap.release()
    cv2.destroyAllWindows()
