"""
1.Build three arrays to store the image in train, in database and not in database files respectively.
2.Build another three arrays to store the name of each image in train, in database and not in database files respectively.
3.use read image function to input all images
4.use comparison function to do comparison
    4-1.Use the random function to choose a tested image in in database or not in database file at a time.
    4-2.Compare the chosen image with all images in train file.
        PS: Use the cosin similarity function to do the comparison
    4-3.Output the comparison result
        PS: The minimum value within all output values will be the closest answer for the chosen image.
"""
import cv2
import os
from pathlib import Path
from scipy.spatial import distance
import random

# read the train images , resize them to 256x256 form and change all to grayscale
# store the image in img_arr
# store the person's name in name_arr
def read_image(img_arr, name_arr,path):
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img_arr.append(img)
        name = Path(filename).stem
        name_arr.append(name)
# function do comparison
def comparison(basic_arr,basic_name_arr,img_arr, name_arr):
    img = random.choice(name_arr)
    index = name_arr.index(img)
    print("Person in chosen image is " + " : " + str(img))
    # do cosin similarity comparison between train images and one of images in indatabase file or not in database file
    x = cosin_Sim(basic_arr, img_arr, index)
    # show the comparison result and do the prediction
    print("================================================================")
    print("Comparison Result")
    print("================================================================")
    for i in range(len(x)):
        print(basic_name_arr[i] + " : " + str(x[i]))
    print("================================================================")
    print("End Comparison")
    print("================================================================")
    min_index = x.index(min(x))
    if(min(x)== 0):
        print("The person in chosen image is : " + str(basic_name_arr[min_index]))
    else:
        print("The person in chosen image most likely be: " + str(basic_name_arr[min_index]))

    print("-------Complete one case----------")


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


# in database image comparison
comparison(image, image_name, indata_img, indata_img_name)
# not in database image comparison
comparison(image, image_name, not_indata_img, not_indata_img_name)