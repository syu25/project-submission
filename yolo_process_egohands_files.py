import scipy.io
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import copy
import csv

# def get_bbox_visualize(base_path, dir):
#
#
# def generate_csv_files(image_dir):
#     for root, dirs, filenames in os.walk(image_dir):
#         for dir in dirs:
#             get_bbox_visualize(image_dir, dir)
#
#     print("CSV generation complete!\nGenerating train/test/eval folders")
#     split_data_test_eval_train("egohands/_LABELLED_SAMPLES/")


#Credit to Victor Dibia for the code to parse egohands matlab file
counter = 0
for root, dirs, filenames in os.walk('./egohands_data/_LABELLED_SAMPLES/'):
    # print('root: {} dirs: {} filenames: {}'.format(root,dirs,filenames))

    base_path = './egohands_data/_LABELLED_SAMPLES/'
    boxes = []
    polygons = []
    image_path_array = []

    for f in filenames:
        if(f.split('.')[1] == "jpg"):
            img_path = root + '/' + f
            image_path_array.append(img_path)

        if f == 'polygons.mat':
            boxes = scipy.io.loadmat(root + '/polygons.mat')
            polygons = boxes['polygons'][0]
    image_path_array.sort()

    pointindex = 0
    #print('len(polygons): {}'.format(len(polygons)))
    for allhands in polygons:
        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)
        raw_img = copy.deepcopy(img)
        plt.imshow(img)

        #cv2.imshow('hi',img)
        img_params = {}
        img_params['width'] = np.size(img,1)
        img_params['height'] = np.size(img,0)
        head, tail = os.path.split(img_id)
        img_params['filename'] = tail
        img_params['path'] = os.path.abspath(img_id)
        img_params['type'] = 'train'
        pointindex+=1

        mask = np.zeros((img_params['height'],img_params['width'],3), dtype=np.uint8)

        boxarray = []
        for hand in allhands:
            pts = np.empty((0,2),int)
            max_x = max_y = min_x = min_y = height = weight = 0

            findex = 0
            for point in hand:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])
                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex+=1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y

                    appeno = np.array([[x,y]])
                    pts = np.append(pts,appeno, axis = 0)
            hold = {}
            hold['min_x'] = min_x
            hold['min_y'] = min_y
            hold['max_x'] = max_x
            hold['max_y'] = max_y
            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                boxarray.append(hold)
                labelrow = [tail, np.size(img, 1), np.size(img, 0), "hand", min_x, min_y, max_x, max_y]
                #csv save file but we need different type
            cv2.polylines(img, [pts], True, (0,255,255),1)
            #NOTE: if there is a crash with C0000005 or some -10..., then array is empty
            red = random.randint(0,255)
            green = random.randint(0,255)
            blue = random.randint(0,255)
            if(len(pts) != 0):
                cv2.fillPoly(img, [pts], [red,green,blue])
                cv2.fillPoly(mask, [pts], [red,green,blue])
            cv2.rectangle(img, (min_x, max_y), (max_x, min_y), (0,255,0), 1)


        #cv2.imshow('verify annotation', img)
        print('counter: {}'.format(counter))
        #cv2.waitKey(0)
        #there are frames with no hands so we need to filter those out
        if np.any(mask):
            cv2.imwrite('./egohands_processed/yolo_images/image{}.png'.format(counter),raw_img)
            #cv2.imwrite('./egohands_processed/masks/mask{}.png'.format(counter),mask)
            #boundary_box = [min_x, min_y, max_x, max_y]
            #label = 1
            boundary_boxes = boxarray
            #May have to give each a label for now I'll just give them all 1 since rcnn is instance segmentation
            #that being said I only have one class so it should be okay
            labels = np.ones(len(boxarray))
            image_id = counter
            area = (max_x-min_x+1)*(max_y-min_y+1)
            iscrowd = False
            #   BOUNDINGBOX, LABEL=1, IMAGE_ID, BOUNDINGBOX_AREA, ISCROWD=False
            #   "###,###,###,### 1 0 12204 False"
            with open('./egohands_processed/yolo_annotations/annotation{}.txt'.format(counter),'w',newline='') as csvfile:
                writer = csv.writer(csvfile,delimiter=' ')
                output = '0'
                for b in range(0,len(boundary_boxes)):
                    output += ' '
                    #output = 'boundary_box' + ' ' + \
                    output += str((boundary_boxes[b]['max_x']+boundary_boxes[b]['min_x'])/2/img_params['width']) + ' ' +\
                                str((boundary_boxes[b]['max_y'] + boundary_boxes[b]['min_y'])/2/img_params['height']) + ' ' +\
                                str((boundary_boxes[b]['max_x'] - boundary_boxes[b]['min_x'])/img_params['width']) + ' '+\
                                str((boundary_boxes[b]['max_y'] - boundary_boxes[b]['min_y'])/img_params['height'])
                    writer.writerow([output])
            counter += 1