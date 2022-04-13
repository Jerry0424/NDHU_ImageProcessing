'''
Q1 Add Salt & Pepper Noise

        To obtain an image with slat & pepper noise,
        we need to add white and black pixels randomly in the image matrix.

In both filters, I adopt the strategy, ignoring the edges to deal with border problem

Q2 Mean Filter
    get the center coordinate and use its neighbors to compute the mean value
    Then, replace the center with mean calculated
Q3 Median Filter
    get the center coordinate and store its neighbors
    After that, sort all store pixels values in ascending order
    Then, replace the center with median value within sorted data
'''

import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# add noise function
def add_Salt_and_Pepper_Noise(img):

    # get the dimension of image
    row, col = img.shape[0], img.shape[1]

    # pick a random number between 300 and 1000 as the number of pixels colored to white
    N_pixels_white = random.randint(300,1000)

    # randomly pick some pixels in the image to color them to white
    for i in range(N_pixels_white):
        # pick a random pixel's coordinate
        pixel_x = random.randint(0, row - 1)
        pixel_y = random.randint(0, col - 1)

        # color the picked pixel to white
        img[pixel_x][pixel_y] = 255

    # pick a random number between 300 and 1000 as the number of pixels colored to black
    N_pixels_black = random.randint(300,1000)

    # randomly pick some pixels in the image to color them to black
    for i in range(N_pixels_black):
        # pick a random pixel's coordinate
        pixel_x = random.randint(0, row - 1)
        pixel_y = random.randint(0, col - 1)

        # color the picked pixel to black
        img[pixel_x][pixel_y] = 0



    return img



# k is the filter size
# mean filter function
def mean_filter(img, k):

    # copy an image to do mean filter
    img_adjust = img.copy()

    # get the dimension of image
    img_row = img.shape[0]
    img_col = img.shape[1]

    # get the coordinate of the staring pixel
    start_x = k // 2
    start_y = k // 2

    # n represents the neighbors range
    # If n = 1  and center coordinate is (x , y), the neighbor range for (x,y) is permutations of x which is in range  x - 1 ~ x + 1 and y which is in range y - 1 ~ y + 1
    n = k // 2

    # compute the mean value to replace the center pixel value
    # the range of rows and columns for target pixels is following
    # row starts from start_x to (img_row - start_row)
    # column starts from start_y to (img_col - start_col)
    for i in range(start_x, img_row - start_x):
        for j in range(start_y, img_col - start_y):
            # compute the mean
            mean = 0
            # sum all the pixel values in neighbor range
            for x in range(-n, n + 1):
                for y in range(-n, n + 1):
                    mean += img_adjust[i + x][j + y]
            # divide the sum by the number of pixels in neighbor range to get mean
            mean /= k * k

            # replace the center
            img_adjust[i][j] = mean

    return img_adjust



# define my sort function to do the sort in median filter
# I use bubble sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp




# median filter function
def median_filter(img, k):
    # copy an image to do median filter
    img_adjust = img.copy()

    # get the dimension of image
    img_row = img.shape[0]
    img_col = img.shape[1]

    # get the coordinate of the staring pixel
    start_x = k // 2
    start_y = k // 2

    # n represents the neighbors range
    # If n = 1  and center coordinate is (x , y), the neighbor range for (x,y) is permutations of x which is in range  x - 1 ~ x + 1 and y which is in range y - 1 ~ y + 1
    n = k // 2

    # put all pixel included in neighbor range in to an array and sort all values in increasing order
    # pick the middle pixel value to replace center pixel value
    # the range of rows and columns for target pixels is following
    # row starts from start_x to (img_row - start_row)
    # column starts from start_y to (img_col - start_col)
    for i in range(start_x, img_row - start_x):
        for j in range(start_y, img_col - start_y):
            # creat the array to store pixel value in neighbor
            middle = [0] * k * k
            # cnt is the index counter to help us store pixel value into middle array
            cnt = 0
            # find the neighbor pixels and store them into middle array
            for x in range(-n, n + 1):
                for y in range(-n, n + 1):
                    middle[cnt] = img_adjust[i + x][j + y]
                    cnt += 1

            # sort all values in middle in increasing order
            bubble_sort(middle)

            # pick the median pixel to replace the center
            img_adjust[i][j] = middle[((k * k) // 2)]

    return img_adjust


# main program


# read the image
img = mpimg.imread('data_ex3.jpg')


# title is used to put the title on output image
# imshow is used to put image on the plot
# savefig is used to save image in project file
# show is used to show the image on plot

# original image
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.savefig('D:\ImageConvert\HW3\Original.jpg')
plt.show()




# add the salt & pepper noise and show adjust image
img_noise = add_Salt_and_Pepper_Noise(img)
plt.title('Adding Salt & Pepper Noise')
plt.imshow(img_noise, cmap='gray')
plt.savefig('D:\ImageConvert\HW3\ noise.jpg')
plt.show()



# mean filter 3 * 3
mean_f_3 = mean_filter(img_noise, 3)
plt.title('3x3 Mean Filter')
plt.imshow(mean_f_3, cmap='gray')
plt.savefig('D:\ImageConvert\HW3\ mean3.jpg')
plt.show()



# mean filter 5 * 5
mean_f_5 = mean_filter(img_noise, 5)
plt.title('5x5 Mean Filter')
plt.imshow(mean_f_5, cmap='gray')
plt.savefig('D:\ImageConvert\HW3\ mean5.jpg')
plt.show()



# median filter 3 * 3
median_f_3 = median_filter(img_noise, 3)
plt.title('3x3 Median Filter')
plt.imshow(median_f_3, cmap='gray')
plt.savefig('D:\ImageConvert\HW3\ median3.jpg')
plt.show()


# median filter 5 * 5
median_f_5 = median_filter(img_noise, 5)
plt.title('5x5 Median Filter')
plt.imshow(median_f_5, cmap='gray')
plt.savefig('D:\ImageConvert\HW3\ median5.jpg')
plt.show()
