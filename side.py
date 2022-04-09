import PIL
from PIL import Image
from PIL import ImageFilter
import cv2
#img=cv2.imread('D:/vs project/dump_a_dump/train1/Image/1 .png',0)
#img1=Image.open('D:/vs project/dump_a_dump/train1/Image/1 .png')

import numpy as np
# 读取原图像

'''
# 显示原图像
cv2.namedWindow('img', 0)
cv2.resizeWindow('img', 400, 600)
cv2.imshow('img', img)
#cv2.waitKey(0)
# 高斯模糊
img_rgb = cv2.GaussianBlur(img, (5, 5), 0)
canny_img = cv2.Canny(img_rgb, 1, 10)

# 显示边缘检测图像
cv2.namedWindow('canny', 0)
cv2.resizeWindow('canny', 400, 600)
cv2.imshow('canny', canny_img)
#cv2.waitKey(0)
# 输出边缘检测图像的高和宽
y_top = np.nonzero([max(row) for row in canny_img[400:]])[0][0] 
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
picture=cv2.imread('D:/vs project/dump_a_dump/current.png')
def jump_peo(picture):
    #img_rgb = cv2.imread('D:/vs project/dump_a_dump/train1/Image/1 .png')
    #pt=[]
    #solve=[]
    img_rgb=picture
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('jump.png',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    global pt
    for pt in zip(*loc[::-1]):
        
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        #solve.append([pt])
    #print(pt)
    cv2.imwrite('res.png',img_rgb)
    center=[pt[0]+w/2,pt[1]+h/2]
    
    return center



