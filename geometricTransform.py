import cv2
import numpy as np

img = cv2.imread('gerald.jpeg',0)


def gamma_transformation(img, gamma):
    '''

    :param img: a image readed with opencv
    :param gamma: a int gama
    :return: a image with the transformation applied
    '''
    img_transformed = np.power(img, gamma)
    return img_transformed

def logaritimic_transformation(img, thresh, px):
    #normalized_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_log = np.uint8(np.log1p(img))
    img_transformed = cv2.threshold(image_log, thresh, px, cv2.THRESH_BINARY)[1]
    return img_transformed

def negative_transformation(img,sub):

    return sub - img


img_1 = negative_transformation(img , 255)
img_2 = gamma_transformation(img, 200)
img_3 = logaritimic_transformation(img, 3, 255)
cv2.imshow('input', img)
cv2.imshow('negative', img_1)
cv2.imshow('logaritimic', img_3)
cv2.imshow('gamma', img_2)




cv2.waitKey(100000)
cv2.destroyAllWindows()

