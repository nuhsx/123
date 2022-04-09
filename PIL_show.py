import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch as t
import PIL
from PIL import Image
from torchvision.datasets import ImageFolder
import os
import numpy as np
import cv2
import copy
'''
#可以把Tensor转化为Image，方便可视化
show = ToPILImage()

#先伪造一个图片的Tensor，用ToPILImage显示
fake_img = t.randn(3, 32, 32)

#显示图片
I=show(fake_img)
I.show()
'''
'''
y_train_path='D:/vs project/dump_a_dump/train_labels/'
def label(path):
    fi=[]
    img_list=os.listdir(path) 
    for i in img_list:
        f = open(f'{path}{i}', "r")
        fi.append(float(f.readlines()[0]))
    return t.tensor(fi)
y_train=label(y_train_path)
print(y_train)
'''
'''
def Image(path):
    fi=[]
    img_list=os.listdir(path) 
    for i in img_list:
        f =  PIL.Image.open(f'{path}{i}')
        #print(f)
        fi.append(np.asarray(f))
    
    return t.tensor(fi)
x_train_path='D:/vs project/dump_a_dump/train/Image/'
print(Image(x_train_path))
'''
img_blur=cv2.imread('D:/vs project/dump_a_dump/current.png')
'''
# X轴方向的Sobel 边缘检测
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) 
# Y轴方向的Sobel 边缘检测
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) 
# XY轴方向的Sobel 边缘检测
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) 
#显示图片
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
def target_peo(img_blur):
# 读入图像
    image=img_blur
    #center=[]
    lenna = img_blur
    # 图像降噪
    lenna = cv2.GaussianBlur(lenna, (5, 5), 0)
    # Canny边缘检测，50为低阈值low，150为高阈值high
    canny = cv2.Canny(lenna, 50, 150)
    c=(np.array(copy.deepcopy(canny)))

    
    #print(len(c))
    first2=[]
    long=len(c[0])
    first=(np.argwhere(c==255))[0]
    #first=[first[0],first[1]]#1是左右
    
    for i in range(300):
        #print(i)
        first2.append(first[1])
        if c[first[0]+i+1][first[1]]==255 and len(first2)==0:
            first2=[]
            first2.append(first[1]+i+1)
            
    #print(first2[0])
    #first2[0][1]
    #for i in range(20):
        #c[first[0],first[1]+i]=255
    
    
    #cv2.imshow("canny", c)
    #print(first)
    center=[(first[0]+first2[0])/2,first[1]]
    #print(center)
    #cv2.waitKey()
    #print(center)
    return center
#target_peo(img_blur)
#target_peo(img_blur)