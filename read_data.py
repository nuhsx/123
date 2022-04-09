import numpy as np
import random
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import n
import net
import PIL
import torch
import torch.nn as nn 
from torch import optim
import numpy as np
from torchvision.datasets import ImageFolder
import os
import _thread
bs=1
steps=200 

#损失方法
loss_func= nn.MSELoss()
#loss_func=nn.CrossEntropyLoss()


#实例化定义的网络结构
model=net.nnet(bs)
model=model.apply(net.init_weights)
save_model=net.nnet
#继续训练

#model=Nnet.nnet(bs)

#model=Nnet.nnet(bs)
#save_model=model
#reload_states = torch.load("state.pt")
#model.load_state_dict(reload_states['net'])



#优化器
optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-7)
#optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-4)
#optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
#optimizer =optim.Adagrad(model.parameters(), lr=0.0001, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000], gamma=0.1)
def time ():
    path='D:/vs project/dump_a_dump/train_data/'
    train=[]
    val=[]
    i_train=[]
    i_val=[]
    dict=np.load("D:/vs project/dump_a_dump/train_data/time.npz")
    j=0

    for  i in dict['abc']:
        
        ran=random.random()
        if ran>0.1:
            train.append(np.array([i]))
            f =  (PIL.Image.open(f'{path}{j}.jpg').convert("L"))
            f=np.array(f)
            i_train.append(np.array(f))
        else:
            val.append(np.array([i]))
            f =  (PIL.Image.open(f'{path}{j}.jpg').convert("L"))
            i_val.append(np.array(f))
        j+=1
    return np.array(train),np.array(val),np.array(i_train),np.array(i_val)
def loss_batch(model,loss_func,yb,xb,optimizer,isnot,long):
    #print(xb.shape)
    #print(yb.shape)
    
    loss=loss_func(model(xb.reshape(-1,1,100,100)),yb.reshape(-1,1))#解决bs问题
    #print(f'train{(model(xb.reshape(-1,1,100,100)))}.item()')
    #print(f'val{(yb.reshape(-1,1))}')
    if len(long)%100==0:
        print('saving')
        state={'net':model.state_dict(),'optimizer':optimizer.state_dict()}#,'epoch':step}
        torch.save(state,f'log/state.pt')#   
        print('save')
        print(state['net'])
    print('当前step：'+str(len(long)),'x:'+str((model(xb.reshape(-1,1,100,100))).item()),'y:'+str((yb.reshape(-1,1))),'损失：'+str(loss.item()))
    #print(loss.item())
    if isnot==True:
        if optimizer is not None:
            loss.requires_grad_(True) 
        
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    return loss.item(),len(xb)
def fit(steps,model,loss_func,optimizer,train_dl,valid_dl):
    '''训练方法
        steps迭代多少次
        model实例化的网络
        loss_func 损失方法
        optimizer 优化器
        train_dl,valid_dl
        #fit的调用 fit(steps,model,loss_func,optimizer,get_data)
    '''
    val=[]
    long=[]
    longs=[]
    best=[]
    xlong=[]
    writer= SummaryWriter("logs")
    #writer1= SummaryWriter("log")
    val_loss=1
    step=0
    while val_loss>0.0001:
        
        train_loss=[]
        #print(f'第{len(long)}个epoch')
        longs.append(0)
        #long=[]
        model=model.train()
        for xb,yb in train_dl:#一次train_dl
            
            long.append(0)
            y,x=loss_batch(model,loss_func,xb,yb,optimizer,True,long)#返回loss和训练集长度
            train_loss.append(y)
            #writer.add_scalar('train',y,len(xlong))
            #print('#'*len(long))
            
        model=model.eval()
        with torch.no_grad():
            
            print('eval')
            losses,nums=zip(
                *[loss_batch(model,loss_func,xb,yb,optimizer,True,long)for xb,yb in valid_dl]
                )#函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            val_loss=np.sum(np.multiply(losses,nums))/np.sum(nums)

            #print('当前step：'+str(step),'验证集损失：'+str(val_loss))
            xlong.append(0)
            #writer.add_scalar('max',val_loss,len(xlong))
            #writer.add_scalar('val',val_loss,len(xlong))
            if step==0:
                #torch.save(save_model, 'best.pt')
                best.append(val_loss)
                #writer.close()
            
        avg=sum(train_loss)/len(train_loss)
        #print()
        #writer.add_scalar('epoch_train',avg,len(longs))
        writer.add_scalars("mix", {'train':avg ,'val': val_loss}, len(longs))
        step+=1
        #writer1.add_scalar('max',avg,len(longs))
    writer.close()
train,val,i_train,i_val=time()
train,i_train,val,i_val=map(torch.tensor,(train,i_train,val,i_val))
train_ds=TensorDataset(train.float(),i_train.float())
valid_ds=TensorDataset(val.float(),i_val.float())
train_dl=DataLoader(train_ds ,batch_size=bs,shuffle=True)#dataloader
valid_dl=DataLoader(valid_ds ,batch_size=int(bs))
fit(steps,model,loss_func,optimizer,train_dl,valid_dl)