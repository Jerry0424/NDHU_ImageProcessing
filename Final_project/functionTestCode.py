"""
The Comparison isn't in function form
"""



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