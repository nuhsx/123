from random import randint
import os
import cv2
from numpy.lib.type_check import imag
import side
from PIL import ImageGrab
import PIL
import pyautogui as auto
import torch
#import tictactoe_ops as game
from pickle import TRUE
import random
import numpy as np
import torch
import torch as t
from torch import nn
from numpy.lib.function_base import average, copy
import torch
from torch.nn.parameter import Parameter
from torch import nn,optim
from torch.functional import Tensor
from torch.nn import functional as F
import numpy as np
import random
import copy
from torchvision import transforms
import n
import PIL_show
import random
import time
import math
#print (torch.cuda.is_available())
import matplotlib
import matplotlib.pyplot as plt
import cv2
'''

a=[]
for i in range(10):
    for j in range(1,100):
        if (6290-j*i)%j==0:
            a.append([i,j])
print(a)
i=0
while i !=10:
    print(i)
    r=randint(0,1)
    print(r)
    if r==1:
        i+=1
        #print(i)
'''
i_p=[]
j_l=[]
'''
for i in img_list:
    i_p.append(i)
#print(i_p)
for j in lab_list:
    j_l.append(j)
    '''
def label(path):
    fi=[]
    img_list=os.listdir(path) 
    for i in img_list:
        f = open(f'{path}{i}', "r")
        fi.append(float(f.readlines()[0]))
    return fi

def Image(path):
    I=[]
    distance_load=[]
    img_list=os.listdir(path) 
    for i in img_list:
        #print(i)
        f =  cv2.imread(f'{path+i}')
        center1=side.jump_peo(f)
        center2=PIL_show.target_peo(f)
        pingfang1=(center2[0]-center1[0])**2
        pingfang2=(center2[1]-center1[1])**2
        distance=math.sqrt(pingfang1+pingfang2)
        distance_load.append(distance)

    #print(distance_load)
    return distance_load

p_path = "D:/vs project/dump_a_dump/photos/" 
l_path="D:/vs project/dump_a_dump/labels/" 
y=np.array(label(l_path)).reshape(-1)#时间数据数组

x =np.array(Image(p_path) ).reshape(-1)
#print(str)

#print(x.shape)
#print(y)
matplotlib.use ('TkAgg')
plt.scatter(x,y)

plt.show()

params=np.polyfit(x,y,100)
print(params)
pa_fu=np.poly1d(params)
y_pre=pa_fu(x)
plt.scatter(x,y)
plt.plot(x,y_pre)
plt.show()
#print(plt.get_backend())


#print(distance)










