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
import random
import time
import os.path 
'''
optimizer = optim.Adam(n.Net().parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.MSELoss()
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1500, 2500], gamma=0.1)
'''
better=[0.5423]
f = open("filename.txt", "r",encoding='utf-8')
for i in f:
    better.append (float(i))
p_path = "D:/vs project/dump_a_dump/photos/" 
l_path="D:/vs project/dump_a_dump/labels/" 

auto.FAILSAFE = True
for i in range (100000):
    time.sleep(1)
    bbox = (0, 0, 470, 870)
    im = ImageGrab.grab(bbox)
    ran=random.random()
    #img_tensor = transforms.ToTensor()(im)
    if ran>0.5:
        R=random.choice(better)
    else:
        R=random.uniform(0.080,1)
    #R=float(s)
    auto.mouseDown(x=200, y=700, button='left')
    time.sleep(R)
    print(R)
    auto.mouseUp()
    time.sleep(2.7)
    if auto.locateOnScreen('stop.png') ==None: 
        print ("没找到")
        lfull_path = l_path + f'{i} .txt'
        pfull_path = p_path + f'{i} .png'
        im.save(pfull_path)
        file = open(lfull_path, 'w')
        file.write(str(R))
        file.close()
        #p,w=n.Net()(img_tensor)
        
        

        '''
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss = criterion(torch.tensor(R).reshape(1,1),p,1,w)                 
        loss.backward()    
        optimizer.step()
        lr_scheduler.step()
        '''
    else:

     
        #p,w=n.Net()(img_tensor)
    
     
        
        x,y,width,height=auto.locateOnScreen('stop.png')
        auto.click(x,y,button='left')
        
        
    #torch.save(n.Net, 'newest2.pt')