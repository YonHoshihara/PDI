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

def logaritimic_transformation(img, thresh, px):
    #normalized_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_log = np.uint8(np.log1p(img))
    img_transformed = cv2.threshold(image_log, thresh, px, cv2.THRESH_BINARY)[1]
    return img_transformed

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

cv2.imshow('input', img)
cv2.imshow('negative', img_1)
cv2.imshow('gamma', img_2)



#histogram(img_2)

#img_3 = logaritimic_transformation(img, 3, 255)
#cv2.imshow('logaritimic', img_3)

# plt.subplot(img.ravel(), 256, [0, 256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(img.ravel(), 256, [0, 256])
# plt.show()
cv2.waitKey(100000)
cv2.destroyAllWindows()

