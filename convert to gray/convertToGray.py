
'''
Transform a color image into gray image using the conversion formula.
Show the pictures using matplotlib.
'''

# use matplotlib to help get the image and show the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# get the color image
img = mpimg.imread('lena.png')

# put the color image into a subplot
plt.subplot(2,1,1)
plt.imshow(img)




# Convert the color image into grayscale using the formula which adjusts the values of RGB
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
imgGray = 0.299 * R + 0.587 * G + 0.114 * B



# put the changed image which now is grayscale into the other subplot
plt.subplot(2,1,2)
plt.imshow(imgGray, cmap='gray')


# show the images
plt.show()


