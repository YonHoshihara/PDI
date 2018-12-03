import cv2
import numpy as np
from matplotlib import pyplot as plt
# O = (i/255)^gamma

def gamma_transformation(img, gamma):
    '''

    :param img: a image readed with opencv
    :param gamma: a int gama
    :return: a image with the transformation applied
    '''
    img_transformed = np.power(img/255, gamma)
    return img_transformed

def logaritimic_transformation(img, c):
    new_image = []

    for element in img:
        pixels = []

        for pixel in element:

            pixel = c*(np.log(1 + pixel))
            pixels.append(pixel)
        new_image.append(pixels)
    new_image = np.array(new_image)
    return new_image

def negative_transformation(img,sub):

    return sub - img

def histogram_color(img):

    color = ('b', 'g', 'r')

    for i, col in enumerate(color):

        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def histogram_bw(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


img = cv2.imread('image.jpeg', 0)
img_color = cv2.imread('image.jpeg')
img_2 = gamma_transformation(img, 0.5)
img_1 = negative_transformation(img, 255)
img_3 = logaritimic_transformation(img, 0.2)

cv2.imshow('input', img)
cv2.imshow('negative', img_1)
cv2.imshow('gamma', img_2)
cv2.imshow('logaritimic', img_3)


#histogram(img_2)



# plt.subplot(img.ravel(), 256, [0, 256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(img.ravel(), 256, [0, 256])
# plt.show()
cv2.waitKey(100000)
cv2.destroyAllWindows()

