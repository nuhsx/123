from PIL import ImageGrab
import PIL
import pyautogui as auto
import torch
#import tictactoe_ops as game
from pickle import TRUE
import net
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
#import n
import net
import random
import time
import os.path 
from PIL import Image
from numpy import average, dot, linalg
'''
optimizer = optim.Adam(n.Net().parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.MSELoss()
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1500, 2500], gamma=0.1)
'''
import Nnet



auto.FAILSAFE = True
for i in range (100000):
    time.sleep(1)
    bbox = (0, 210, 430, 640)
    
    
    #img_tensor = transforms.ToTensor()(im)
    im = (ImageGrab.grab(bbox))
    im=im.convert("L")#.resize((100,100))
    im.save('current.png')
    '''
    choose=[]
    image1 = im
    img_list=os.listdir('D:/vs project/dump_a_dump/train/Image/') 
    for i in img_list:
        image2 =  PIL.Image.open(f'{x_train_path}{i}')
        cosin = similiar(image1, image2)
        print('图片余弦相似度',cosin)
        if cosin>=0.9:
            #选择标签
            break
    '''
    
    img_tensor = transforms.ToTensor()(im)
    #model=torch.load('best.pt')
    model=Nnet.nnet(1)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

    #model=model.eval()
    reload_states = torch.load("state.pt")
    torch.no_grad()
    model.load_state_dict(reload_states['net'])
    print(reload_states['net'])
    
    #optimizer.load_state_dict(reload_states['optimizer'])
    #print(img_tensor.shape)
    R=model(img_tensor.reshape(1,1,430,430))
    #R=float(s)
    auto.mouseDown(x=200, y=700, button='left')
    print(R)
    time.sleep(R.item())
    
    auto.mouseUp()
    time.sleep(3.5)
    if auto.locateOnScreen('stop.png') ==None: 
        print ("没找到")
    else:

     
        #p,w=n.Net()(img_tensor)
    
     
        
        x,y,width,height=auto.locateOnScreen('stop.png')
        auto.click(x,y,button='left')
'''
def similiar(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res
 '''
 



