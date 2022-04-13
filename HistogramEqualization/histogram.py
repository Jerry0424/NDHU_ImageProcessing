import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np




# function change color to gray

def color2gray(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgray = 0.299 * R + 0.587 * G + 0.114 * B
    return imgray




# function do the histogram equalization process

def HistogramEqualization(img):

    # change image array into one dimension
    img_matrix = img.flatten()

    # get the changed image's gray level int value since its value has been changed to float type in color2gray function
    for i in range(0,len(img_matrix)):
        img_matrix[i] = round(img_matrix[i], 0)

    #change the image's matrix type from float to int
    img_matrix = np.array(img_matrix, dtype='int64')



    # show original gray photo and it's histogram
    img_round = np.array(img_matrix).reshape(img.shape[0], img.shape[1])
    plt.subplot(221)
    plt.imshow(img_round, cmap='gray')
    plt.subplot(222)
    plt.hist(img_round.ravel(), 256, [0,256])



    # make a matrix to record the table of the numbers ni of gray values
    matrix = [0] * 256

    # count the numbers ni of gray values from gray level 0 ~ 255
    for i in range(0,len(img_matrix)):
        matrix[img_matrix[i]] += 1

    # make a matrix to record the cumulative numbers ni of gray values
    cul_matrix = matrix.copy()

    # do the cumulative process from gray level 0 ~ 255
    for i in range(1,len(cul_matrix)):
        cul_matrix[i] = cul_matrix[i - 1] + cul_matrix[i]




    # apply the equation (n0 + n1 + ..... +ni) * (L - 1) / n
    # cul_matrix[img_matrix[i]] represent cumulative value of gray level which stored in img_matrix[i]
    # 256 - 1 = L - 1
    # cul_matrix[255] = n
    for i in range(0, len(img_matrix)):
        img_matrix[i] = round(cul_matrix[img_matrix[i]] * (256-1) / cul_matrix[255], 0)



    # show the chaned gray photo and it's histogram
    img_HE_arr = np.array(img_matrix).reshape(img.shape[0], img.shape[1])
    plt.subplot(223)
    plt.imshow(img_HE_arr, cmap='gray')
    plt.subplot(224)
    plt.hist(img_HE_arr.flatten(), 256, [0, 256], color='red')





# read the targeted images
img = mpimg.imread('Bear.jpg')
img1= mpimg.imread('Bear1.jpg')
img2= mpimg.imread('Bear2.jpg')

img_arr= {}
img_arr[0] = img
img_arr[1] = img1
img_arr[2] = img2



# change  images into gray
for i in range(0,len(img_arr)):
    img_arr[i] = color2gray(img_arr[i])


# Do the Histogram Equalization on each image and show the result respectively
for i in range(0,len(img_arr)):
    HistogramEqualization(img_arr[i])
    plt.show()








