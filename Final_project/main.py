import cv2
import os
from pathlib import Path


# read the train image
# store the image in img_arr
# store the person's name in name_arr
def read_image(img_arr, name_arr,path):

    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename)
        img_arr.append(img)
        name = Path(filename).stem
        name_arr.append(name)


# read the train image
image = []
image_name = []
path = "data/train"
read_image(image, image_name, path)


# read the indatabase image
path = "data/test/indatabase"
indata_img = []
indata_img_name = []
read_image(indata_img, indata_img_name, path)


# read the not indatabase image
path = "data/test/not in database"
not_indata_img = []
not_indata_img_name = []
read_image(not_indata_img, not_indata_img_name, path)


for i in range(len(not_indata_img_name)):
    print(not_indata_img_name[i])


