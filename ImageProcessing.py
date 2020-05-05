import os
import io
import cv2
from skimage import io

def pad(src, amount = 12):
    #My training data is 200x200, resnet needs 224, so I pad with 12 on each side, constant border, and white
    return cv2.copyMakeBorder(src, amount, amount, amount, amount, cv2.BORDER_CONSTANT,None, [255,255,255])


# image_names = os.listdir('.\\data\\training_images\\combined')
# image_names.sort()
# image_names_only = image_names
# image_names = [os.path.join('.\\data\\training_images\\combined', img_name) for img_name in image_names]
#
# for i in range(len(image_names)):
#     image = io.imread(image_names[i])
#     image = pad(image, 12)
#     out = cv2.imwrite('./data/training_images/combined_with_padding/{}'.format(image_names_only[i]), image)

image_names = os.listdir('.\\data\\testing_images\\combined')
image_names.sort()
image_names_only = image_names
image_names = [os.path.join('.\\data\\testing_images\\combined', img_name) for img_name in image_names]

for i in range(len(image_names)):
    image = io.imread(image_names[i])
    image = pad(image, 12)
    out = cv2.imwrite('./data/testing_images/combined_with_padding/{}'.format(image_names_only[i]), image)