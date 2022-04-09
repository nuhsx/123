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
import keyboard

import side
import cv2
center1=[]
R=0
dict={}
p_path = "D:/vs project/dump_a_dump/photos/" 
l_path="D:/vs project/dump_a_dump/labels/" 
#k=0.0084269
#k=0.0079
bbox = (0, 210, 430, 640)

class Fit_model(t.nn.Module):
    def __init__(self):
        super(Fit_model,self).__init__()
        self.linear1 = t.nn.Linear(1,16)
        self.relu = t.nn.ReLU()
        self.linear2 = t.nn.Linear(16,1)

        self.criterion = t.nn.MSELoss()
        self.opt = t.optim.SGD(self.parameters(),lr=0.01)
    def forward(self, input):
        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        y = y.squeeze(-1)

        return y

for i in range(1,100000):

    ran=random.random()
    if ran<0.5:
        p_path = "D:/vs project/dump_a_dump/train4/Image/" 
        l_path="D:/vs project/dump_a_dump//train4/labels/" 
    else:
        p_path = "D:/vs project/dump_a_dump/val4/Image/" 
        l_path="D:/vs project/dump_a_dump//val4/labels/" 
    time.sleep(2)
    #im = ImageGrab.grab(bbox)
    
    im = ImageGrab.grab(bbox)
    
    print('请输入')
    
    #im.save('current.png')
    #f =  cv2.imread('current.png')
    keyboard.wait('f')
    #center1=side.jump_peo(f)
    
    print('center1')
    center1=auto.position()
    keyboard.wait('f')
    print('center2')
    center2=auto.position()

    #center2=PIL_show.target_peo(f)
    pingfang1=(center2[0]-center1[0])**2
    pingfang2=(center2[1]-center1[1])**2
    distance=math.sqrt(pingfang1+pingfang2)

    print(distance)
    if center1[1]<center2[1]:
        R=distance*0.0019
    elif center1[1]>center2[1]:
        R=distance*0.00295
        #R=distance*0.00283
    
 
    auto.mouseDown(x=200, y=700, button='left')
    print(R)

    time.sleep(R)
    
    auto.mouseUp()
    lfull_path = l_path + f'{i+576} .txt'
    pfull_path = p_path + f'{i+576} .png'
    im.save(pfull_path)
    
    file = open(lfull_path, 'w')
    file.write(str(R))
    file.close()
    '''
    time.sleep (3.5)
    
    if auto.locateOnScreen('stop.png') ==None :
        print ("没找到")

    else:
	    x,y,width,height=auto.locateOnScreen('stop.png') 
	    print ("该图标在屏幕中的位置是：X={},Y={}，宽{}像素,高{}像素".format(x,y,width,height))

	    auto.click(x,y,button='left')

 '''
    

