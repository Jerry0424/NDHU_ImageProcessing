'''
1. change the given image to grayscale. Thus,image scale will be converted from (512,512,3) to (512,512)
    solution 1 : use convert_to_gray function

Following steps are packed in transform function

    2. get the specific 8x8 field on image to do dct processing
        solution 2:  use get8x8matrix function to divide the submatrix
    3. do dct processing
        solution 3: use dct + dct processing functions
    4 quantization
        solution 4: use quantization function
    5 dequantization
        solution 5: use dequantization function
    6. do dct processing
        solution 6: use idct + idct processing functions
    7. save the image
         solution 7: use saveImage function
    8. count psnr
        solution 8: use psnr function in which we follow the psnr formula to compute the psnr value
'''


import numpy as np
import cv2
import math
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



def convert_to_gray(img):
    # Convert the color image into grayscale using the formula which adjusts the values of RGB
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.299 * R + 0.587 * G + 0.114 * B
    return imgGray


def get8x8matrix(img,row_num, col_num):
    # count the start point of row(i) and col(j) and end boundary row_bound and col_bound
    row_bound = row_num * 8
    col_bound = col_num * 8

    # create a 8x8 matrix to store the targeted submatrix of origin image and use it do dct processing
    dct = np.zeros((8,8))

    # copy the targeted 8x8 field on original image to a submatrix
    dct_i = 0
    for i in range(row_bound - 8,row_bound):
        dct_j = 0
        for j in range(col_bound - 8, col_bound):
            dct[dct_i][dct_j] = img[i][j]
            dct_j += 1
        dct_i += 1

    return dct

# use quantization list to do quantization
def quantization(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = round(matrix[i][j] / q[i][j])

    return matrix

# use quantization list to do dequantization
def dequantization(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] *= q[i][j]

    return matrix



# whole dct process -> function dct + function dctProcess
def dct(matrix):
    # set a matrix to store values computed during the dct processing
    dct = np.zeros((8, 8))

    # subtract 128 for avoiding overflow
    matrix -= 128

    # compute the dct value
    for u in range(8):
        for v in range(8):
            dct[u][v] = dctProcess(matrix, u, v)


    return dct


def dctProcess(matrix, u, v):
    val = 0

    # compute the dct formula
    for x in range(8):
        tmp = 0
        for y in range(8):
            tmp = matrix[x][y]
            tmp *= math.cos(math.pi * (2 * x + 1) * u / 16)
            tmp *= math.cos(math.pi * (2 * y + 1) * v / 16)
            val += tmp


    # c represent C(u) * C(v)
    c = 1
    # check C(u)
    if u == 0:
        c /= np.sqrt(2)
    # check C(v)
    if v == 0:
        c /= np.sqrt(2)


    return val * (2. / np.sqrt(8 * 8)) * c

# whole idct process -> function idct + function idctProcess
def idct(dct):
    idct = np.zeros((8, 8))

    for x in range(8):
        for y in range(8):
            idct[x][y] = idctProcess(dct, x, y)

    # plus 128 back since we subtract 128 before doing dct processing
    idct += 128
    return idct


def idctProcess(dct, x, y):
    val = 0

    # compute the idct formula
    for u in range(8):
        for v in range(8):
            tmp = dct[u][v]

            # compute C(u) * C(v)
            if u == 0:
                tmp /= np.sqrt(2)
            if v == 0:
                tmp /= np.sqrt(2)

            tmp *= math.cos(math.pi * (2 * x + 1) * u / 16)
            tmp *= math.cos(math.pi * (2 * y + 1) * v / 16)
            val += tmp

    return val * (2. / np.sqrt(8 * 8))

def adjust(img, row_num, col_num, matrix):
    # count the start point of row(i) and col(j) and end boundary row_bound and col_bound
    row_bound = row_num * 8
    col_bound = col_num * 8

    matrix_i = 0
    for i in range(row_bound - 8, row_bound):
        matrix_j = 0
        for j in range(col_bound - 8, col_bound):
            img[i][j] = matrix[matrix_i][matrix_j]
            matrix_j += 1
        matrix_i += 1

    return img


def saveImage(dct, idct, error):
    # save dct image
    cv2.imwrite('dct_transform.jpg', dct)
    # save idct image
    cv2.imwrite('idct.jpg', idct)
    # save error image
    cv2.imwrite('error_image.jpg',error)




def transform(img):
    # devide the origin image into several 8x8 matrix to do dct processing
    col,row = img.shape[1],img.shape[0]
    # get the maximum number of the submarix in row and column
    col_num = int(col / 8)
    row_num = int(row / 8)
    # copy the original image to help us doing dct processing instead of using original image directly
    img_copy = img.copy()
    img_copy_2 = img.copy()
    # creat zero 512x512 matrix to save values computed during dct processing
    adjust_img = np.zeros((row, col))
    dct_img = np.zeros((row, col))

    # make each 8x8 submatrix  do dct processing one by one
    for i in range(row_num):
        for j in range(col_num):
            # get the targeted 8x8 submatrix to do dct processing
            matrix = get8x8matrix(img,i+1, j+1)
            # do dct processing
            matrix = dct(matrix)
            dct_img = adjust(img_copy, i + 1, j + 1, matrix)
            # quantize
            matrix = quantization(matrix)
            # dequantize
            matrix = dequantization(matrix)
            # do idct processing
            matrix = idct(matrix)
            # get the adjust image numpy array
            adjust_img = adjust(img_copy_2,i+1, j+1, matrix)

    adjust_img = np.round(adjust_img)
    dct_img = np.round(dct_img)

    # save the asked image such as dct image , idct image , error image
    saveImage(dct_img, adjust_img, img - adjust_img)
    # count the psnr value
    psnr(img, adjust_img)

# compute the psnr formula
def psnr(img, adjust_img):
    mse = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp = (img[i][j] - adjust_img[i][j])
            mse += np.power(tmp, 2)
    mse /= (img.shape[0] * img.shape[1])

    Max = 255 * 255
    val = 10 * np.log10((Max / mse))
    print("PSNR val is :",val)


# read the photo
img = cv2.imread('lena.jpg')
# convert to gray
img = convert_to_gray(img)
# Do all process
transform(img)



















