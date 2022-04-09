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
#import n
import net
import random
import time
import os.path 
from PIL import Image
import net
from numpy import average, dot, linalg
'''
optimizer = optim.Adam(n.Net().parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.MSELoss()
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1500, 2500], gamma=0.1)
'''



auto.FAILSAFE = True

time.sleep(1)
#bbox = (0, 210, 430, 640)
    
    
#img_tensor = transforms.ToTensor()(im)

im = PIL.Image.open('D:/vs project/dump_a_dump/train2/Image/994 .png')
im=im.convert("L").resize((100,100))
img_tensor = transforms.ToTensor()(im)
#model=torch.load('best.pt')
model=net.nnet(1)
reload_states = torch.load("log/state.pt")

model.load_state_dict(reload_states['net'])
#model=model
R=model(img_tensor.reshape(-1,1,100,100))
print(R)
    
    
    #time.sleep(100)

#im.save('current.png')
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
