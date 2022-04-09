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

from torch.utils.tensorboard import SummaryWriter 

import random
import time
import os.path 

 
#print (torch.cuda.is_available())
better=[0.5]
f = open("filename.txt", "r",encoding='utf-8')
for i in f:
    better.append (float(i))
class Net(nn.Module):
    def __init__(self):#参数要改
        super().__init__()

        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)#??? 1,64,109,109
        #self.conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)#输入的通道数=输出，不改变
        self.linear=nn.Linear(26169600,1)
    def forward(self,x):
        out =self.conv1(x)#1t
        #print(out.shape)
        out=self.bn1(out)
        out=F.relu(out)
        out = out.view(1,26169600)
        out =self.linear(out)
        out= F.relu(out)
        return out
bbox = (0, 0, 470, 870)
optimizer = optim.Adam(Net().parameters(), lr=0.1, weight_decay=1e-4)
criterion = nn.MSELoss()
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1500, 2500], gamma=0.1)



for i in range(1000):
    print(i)
    
    old=[]
    new=[]
    time.sleep(2)
    im = ImageGrab.grab(bbox)
    
    #print(stop)    
    # 参数 保存截图文件的路径
    im.save('current.png')
    img_tensor = transforms.ToTensor()(im)
    
    ran=random.random()
    print(img_tensor)
    if ran>0.5:
        R=random.choice(better)
    else:
        R=random.uniform(0.07,1)
    
    #print(a)
    auto.mouseDown(x=200, y=700, button='left')


    time.sleep(R)
    print(R)
    auto.mouseUp()
    time.sleep (3.5)
    
    if auto.locateOnScreen('stop.png') ==None: 
        
        better.append(R)
        im1 = ImageGrab.grab(bbox)
        im1.save('current1.png')
        img_tensor1 = transforms.ToTensor()(im1)
        
        picture_time=Net()(img.reshape(2,3,470, 870))
        file = open('filename.txt','a')
        
        file.write(str(R)+'\n')
        file.close()
        print ("没找到")
        
            
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        p=torch.tensor(R).reshape(1,1)
        loss = criterion(p,picture_time)  
        print(f'loss {p-picture_time}')
        

        loss.backward()    
        optimizer.step()
        lr_scheduler.step()
    else:
	    x,y,width,height=auto.locateOnScreen('stop.png') 
	    print ("该图标在屏幕中的位置是：X={},Y={}，宽{}像素,高{}像素".format(x,y,width,height))
	    #左键点击屏幕上的这个位置
	    auto.click(x,y,button='left')
    if i ==100 or i ==200 or i==400 or i==600 or i ==800:
        print('use net')
        if os.path.isfile('best.pt')==True:
            while auto.locateOnScreen('stop.png') ==None: 
                old.append(0)
                im = ImageGrab.grab(bbox)
                img_tensor = transforms.ToTensor()(im)
                model=torch.load('best.pt')
        #print(model()(img_tensor.reshape(-1,3,408900,1)))
                s=model()(img_tensor.reshape(-1,3,408900,1)) 
                R=float(s)
                auto.mouseDown(x=200, y=700, button='left')
                time.sleep(R)
                print(R)
                auto.mouseUp()
                time.sleep (3.5)
            x,y,width,height=auto.locateOnScreen('stop.png')
            auto.click(x,y,button='left')
            while auto.locateOnScreen('stop.png') ==None: 
                new.append(0)
                im = ImageGrab.grab(bbox)
                img_tensor = transforms.ToTensor()(im)
                s=Net()(img_tensor.reshape(-1,3,408900,1)) 
                R=float(s)
                auto.mouseDown(x=200, y=700, button='left')
                time.sleep(R)
                print(R)
                auto.mouseUp()
                time.sleep (3.5)
            x,y,width,height=auto.locateOnScreen('stop.png')
            auto.click(x,y,button='left')
            if len(new)>len(old):
                torch.save(Net, 'best.pt')
        else:

            torch.save(Net, 'best.pt')


torch.save(Net, 'newest.pt')

            # 计算损失

    
'''
import os
import sys
class Logger():
    def __init__(self, filename="log.txt"): 
        self.terminal = sys.stdout        
        self.log = open(filename, "w")    
    def write(self, message):        
        self.terminal.write(message)      
        self.log.write(message)    
    def flush(self):        
        passsys.stdout = Logger()
        print("Jack Cui")
        print("https://cuijiahua.com")
        print("https://mp.weixin.qq.com/s/OCWwRVDFNslIuKyiCVUoTA")
'''