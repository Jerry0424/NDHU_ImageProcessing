import cv2
import os
from pathlib import Path
from scipy.spatial import distance
import random

# read the train images , resize them to 160x160 form and change all to grayscale
# store the image in img_arr
# store the person's name in name_arr
def read_image(img_arr, name_arr,path):
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img_arr.append(img)
        name = Path(filename).stem
        name_arr.append(name)

def cosin_Sim(image, img, index):
    res = []

    for i in range(len(image)):
        res.append(distance.cosine(image[i].flatten(), img[index].flatten()))

    return res



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


index = random.randint(0,4)
print(index)
print(indata_img_name[index])
x = cosin_Sim(image, indata_img, index)

for i in range(len(x)):
    print(image_name[i] + " : "+ str(x[i]))