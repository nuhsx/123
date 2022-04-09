from PIL import ImageGrab
import PIL
import keyboard
import pyautogui as auto
import torch
#import tictactoe_ops as game
from pickle import TRUE
import random
import numpy as np
import torch
import Nnet
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
import side
import cv2
center1=[]
R=0
model=Nnet.nnet(1)
p_path = "D:/vs project/dump_a_dump/photos/" 
l_path="D:/vs project/dump_a_dump/labels/" 
loss_func= nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000], gamma=0.1)
reload_states = torch.load("state.pt")
model.load_state_dict(reload_states['net'])
optimizer.load_state_dict(reload_states['optimizer'])
bbox = (0, 210, 430, 640)
def loss_batch(model,loss_func,xb,yb,optimizer):
    print(xb.shape)
    loss=loss_func(model(xb.reshape(-1,1,430,430)),yb.reshape(-1,1))#解决bs问题
    print(f'train{(model(xb.reshape(-1,1,430,430)))}.item()')
    print(f'val{(yb.reshape(-1,1))}')
    
    print(loss.item())
    if optimizer is not None:
        loss.requires_grad_(True) 
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    return loss.item(),len(xb)
for i in range(1,10000):
    time.sleep(1.8)
    im = ImageGrab.grab(bbox).convert("L")
    #out = im.convert("L")
    img_tensor = transforms.ToTensor()(np.array(im))
    #model=torch.load('best.pt')
    ran=random.random()
    #print(stop)    
    # 参数 保存截图文件的路径
    #im.save('current.png')
    #f =  cv2.imread('current.png')
    '''
    if ran>0.2:
        reload_states = torch.load("log/state.pt")
        model.load_state_dict(reload_states['net'])
        optimizer.load_state_dict(reload_states['optimizer'])
        print(img_tensor.shape)
        R=model(img_tensor.reshape(-1,1,100,100))
        R_tensor=R
        R=R.item()
    ''' 
    
    print('请输入')
    keyboard.wait('f')
    center1=auto.position()
    print('center1')
    keyboard.wait('f')
    print('center2')
    center2=auto.position()
    pingfang1=(center2[0]-center1[0])**2
    pingfang2=(center2[1]-center1[1])**2
    distance=math.sqrt(pingfang1+pingfang2)

    #print(distance)
    if center1[1]<center2[1]:
        R=distance*0.0019
    elif center1[1]>center2[1]:
        R=distance*0.00295
    #R=random.uniform(0.080,1)
    R_tensor=torch.tensor(np.array([[R]])).float()
    print(R_tensor)
    #center1=side.jump_peo(f)
        
   
    #
    #center2=side.target_peo(f)
    #center2=PIL_show.target_peo(f)
    #pingfang1=(center2[0]-center1[0])**2
    #pingfang2=(center2[1]-center1[1])**2
    #distance=math.sqrt(pingfang1+pingfang2)
    #img_tensor = transforms.ToTensor()(im)
    #print(distance)

    '''
    else:
        im = im.resize((100,100))
        im=im.convert("L")
        im.save('current.png')
        img_tensor = transforms.ToTensor()(im)
        model=torch.load('last.pt')
        model=model(1).eval()
        R=model(img_tensor).item()*1.04
        #R=distance*0.0092
    '''
    
    
    #print(model()(img_tensor.reshape(-1,3,408900,1)))
    #s=model()(img_tensor.reshape(-1,3,408900,1)) 
    #R=random.uniform(0.085,1)

    
    #print(a)
    auto.mouseDown(x=200, y=700, button='left')


    time.sleep(R)
    print(R)
    auto.mouseUp()
    time.sleep (5)
    
    if auto.locateOnScreen('stop.png') ==None: 
        print ("没找到")
        yb=R_tensor
        xb=img_tensor
        loss_batch(model,loss_func,xb,yb,optimizer)
        state={'net':model.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(state,'state.pt')#   
        #lfull_path = l_path + f'{i} .txt'
        #pfull_path = p_path + f'{i} .png'
        #im.save(pfull_path)
        #file = open(lfull_path, 'w')
        #file.write(str(R))
        #file.close()
        '''
        d_file=open('distance.txt','w')
        d_file.write(str(distance))
        d_file.close()
        d_file=open('time.txt','w')
        d_file.write(str(R))
        d_file.close()
        '''
        
    else:
	    x,y,width,height=auto.locateOnScreen('stop.png') 
	    print ("该图标在屏幕中的位置是：X={},Y={}，宽{}像素,高{}像素".format(x,y,width,height))
	    #左键点击屏幕上的这个位置
	    auto.click(x,y,button='left')


            # 计算损失
 
    

