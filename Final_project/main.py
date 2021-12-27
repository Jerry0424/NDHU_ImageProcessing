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

# do the cosin similarity
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

# in database check
index = random.randint(0, 4)
print("Chosen image in database is " + " : " + str(indata_img_name[index]))
# do cosin similarity comparison between train images and one of images in indatabase file
x = cosin_Sim(image, indata_img, index)
# show the comparsion result and do the prediction
print("================================================================")
print("Comparison Result")
print("================================================================")
for i in range(len(x)):
    print(image_name[i] + " : " + str(x[i]))
print("================================================================")
print("End Comparison")
print("================================================================")
min_index = x.index(min(x))
print("The person in chosen image is : " + str(image_name[min_index]))

print("----------------------------------------------------------------")


# not in database check
index = random.randint(0, 4)

print("Chosen image not in database is " + " : " + str(not_indata_img_name[index]))
# do cosin similarity comparison between train images and one of images in not in database file
y = cosin_Sim(image, not_indata_img, index)
# show the comparsion result and do the prediction
print("================================================================")
print("Comparison Result")
print("================================================================")
for i in range(len(y)):
    print(image_name[i] + " : " + str(y[i]))
print("================================================================")
print("End Comparison")
print("================================================================")
min_index = y.index(min(y))
print("The person in chosen image most likely be: " + str(image_name[min_index]))