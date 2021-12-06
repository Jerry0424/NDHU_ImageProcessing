import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# ok
def convert_to_gray(img):
    # Convert the color image into grayscale using the formula which adjusts the values of RGB
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.299 * R + 0.587 * G + 0.114 * B
    return imgGray

def devide_photo(img):

    return






# ok
def DCT(matrix):
    row = matrix.shape[0]
    col = matrix.shape[1]
    # set a dct matrix to store values computed by the dct processing
    dct = np.zeros_like(matrix)

    for i in range(row):
        for j in range(col):
            dct[i, j] = DCT_process(matrix, i, j)

    return dct



# reference need to change
def DCT_process(matrix, i, j):
   width = matrix.shape[1]
   height = matrix.shape[0]
   value = 0.
   for col in range(height):
       for row in range(width):
           save = matrix[col, row]
           save *= math.cos(math.pi * (2 * col + 1) * i / (2. * height))
           save *= math.cos(math.pi * (2 * row + 1) * j / (2. * width))
           value += save
   c = 1.
   if i == 0:
       c /= np.sqrt(2)
   if j == 0:
       c /= np.sqrt(2)

   return (2. / np.sqrt(height * width)) * c * value




# read the photo
img = cv2.imread('lena.jpg')
# convert to gray
img = convert_to_gray(img)
devide_photo(img)



# Quantization List
q = [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
      ]





