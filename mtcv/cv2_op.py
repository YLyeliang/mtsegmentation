import cv2
import os
from mtcv import histEqualize,resize

def minAreaRect(contours):
    """given a set of contours, return minArea Rectangles"""
    rect_keep = []
    for i in contours:
        rect = cv2.minAreaRect(i)
        rect_keep.append(rect)
    return rect_keep

def findContours(img,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(img,mode,method)

def remove_contours(contours,min=600,max=1000):
    """given a set of contours, preserve those of areas greater than min and smaller than max."""
    contours_keep = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > min and area < max:
            contours_keep.append(contours[i])
    return contours_keep

path="D:/dataset/start_end"
imgs= os.listdir(path)
images=[]
for img in imgs:
    images.append(os.path.join(path,img))

img="D:/dataset/Camera_pred/4_Line17_up_20190411032509_29_34km+485.0m_forward.jpg"

for img in images:

    image=cv2.imread(img)
    image=resize(image,ratio=0.3)
    img=image.copy()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # img=resize(img,ratio=0.3)
    cv2.imshow("source",img)
    img2=histEqualize(img,clipLimit=2)

    canny=cv2.Canny(img2,50,250)

    se=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # dst = cv2.morphologyEx(img2,cv2.MORPH_GRADIENT,se)
    # cv2.imshow('gradient',dst)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    ret, binary = cv2.threshold(img2,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # binary =cv2.morphologyEx(binary,cv2.MORPH_OPEN,se)
    # binary=cv2.morphologyEx(binary,cv2.MORPH_ERODE,se)
    # binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,se)
    cnt_img,contours,hierachy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=remove_contours(contours)
    # rect=cv2.minAreaRect(contours)
    cv2.imshow("binary",binary)
    image=cv2.drawContours(image,contours,-1,color=(0,0,255),thickness=1)

    cv2.imshow("clahe",img2)
    cv2.imshow("contour",image)
    cv2.waitKey()