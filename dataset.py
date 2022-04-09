#from torch.functional import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import n
import Nnet
import net
import PIL
import torch
import torch.nn as nn 
from torch import optim
import numpy as np
from torchvision.datasets import ImageFolder
import os
import _thread

'''你的数据--------------------------------------------------------------------------------------'''
x_train_path='D:/vs project/dump_a_dump/val4/Image/'
y_train_path='D:/vs project/dump_a_dump/val4/labels/'
x_valid_path='D:/vs project/dump_a_dump/train4/Image/'
y_valid_path='D:/vs project/dump_a_dump/train4/labels/'
'''超参数-----------------------------------------------------------------------------------'''
picture_size=430
#batch_size一次取数据数
bs=10#需要设一个reshape(-1,x,x,x)
#迭代次数
steps=200 

#损失方法
loss_func= nn.MSELoss()
#loss_func=nn.CrossEntropyLoss()


#实例化定义的网络结构
model=Nnet.nnet(bs)
model=model.apply(Nnet.init_weights)
save_model=Nnet.nnet
#继续训练

#model=Nnet.nnet(bs)

#model=net.nnet(bs)
#save_model=model
#reload_states = torch.load("state.pt")
#model.load_state_dict(reload_states['net'])



#优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
#optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-4)
#optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
#optimizer =optim.Adagrad(model.parameters(), lr=0.0001, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000], gamma=0.1)

def label(path):
    fi=[]
    img_list=os.listdir(path) 
    for i in img_list:
        f = open(f'{path}{i}', "r")
        fi.append(float(f.readlines()[0]))
    return fi

def Image(path):
    I=[]
    img_list=os.listdir(path) 
    for i in img_list:
        f =  (PIL.Image.open(f'{path}{i}').convert("L")).resize((picture_size,picture_size))
        #print(f)
        I.append(np.array(f))#array([],[]) ,array([],[]), array([],[])
    
    return np.array(I)#array([[],[],[],[].......])可以直接给tensor
def get_data(train_ds,valid_ds,bs):
    '''返回Dataloader
    '''
    train_dl=DataLoader(train_ds ,batch_size=bs,shuffle=True)#dataloader
    valid_dl=DataLoader(valid_ds ,batch_size=int(bs))#验证集一次2*bs    valid_dl=DataLoader(valid_ds ,batch_size=bs*2)
    return train_dl,valid_dl
def loss_batch(model,loss_func,xb,yb,optimizer,isnot):
    
    loss=loss_func(model(xb.reshape(-1,1,picture_size,picture_size)),yb.reshape(-1,1))#解决bs问题
    print(f'train{(model(xb.reshape(-1,1,picture_size,picture_size)))}.item()')
    print(f'val{(yb.reshape(-1,1))}')
    
    print(loss.item())
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
    for step in range(steps):
        train_loss=[]
        print(f'第{len(long)}个epoch')
        longs.append(0)
        long=[]
        model.train()
        for xb,yb in train_dl:#一次train_dl
            
            long.append(0)
            y,x=loss_batch(model,loss_func,xb,yb,optimizer,True)#返回loss和训练集长度
            train_loss.append(y)
            #writer.add_scalar('train',y,len(xlong))
            print('#'*len(long))
            
        model.eval()
        with torch.no_grad():
            print('eval')
            losses,nums=zip(
                *[loss_batch(model,loss_func,xb,yb,optimizer,True)for xb,yb in valid_dl]
                )#函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            val_loss=np.sum(np.multiply(losses,nums))/np.sum(nums)

            print('当前step：'+str(step),'验证集损失：'+str(val_loss))
            xlong.append(0)
            #writer.add_scalar('max',val_loss,len(xlong))
            #writer.add_scalar('val',val_loss,len(xlong))
            if step==0:
                #torch.save(save_model, 'best.pt')
                best.append(val_loss)
                #writer.close()
            elif step !=0:
                if val_loss < best[0]:
                    print(best)
                    state={'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':step}
                    torch.save(state,'state.pt')#                                                    log里
                    best=[]
                    best.append(val_loss)
        avg=sum(train_loss)/len(train_loss)
        
        #writer.add_scalar('epoch_train',avg,len(longs))
        writer.add_scalars("mix", {'train':avg ,'val': val_loss}, len(longs))

        #writer1.add_scalar('max',avg,len(longs))
    writer.close()
    


'''---------------------------------------------------------------------------------------------'''

#x_train=ImageFolder(x_train_path,transform=np.numpy())
x_train=Image(x_train_path)
y_train=label(y_train_path)
#x_valid=ImageFolder(x_valid_path,transform=np.numpy())
x_valid=Image(x_valid_path)
y_valid=label(y_valid_path)
'''----------------------------------------------------------------------------------------------'''
x_train,y_train,x_valid,y_valid=map(torch.tensor,(x_train,y_train,x_valid,y_valid))

#训练集
train_ds=TensorDataset(x_train.float(),y_train.float())#训练集dataset
#验证集
valid_ds=TensorDataset(x_valid.float(),y_valid.float())
print('ok')
'''训练方法：
train_dl,valid_dl=get_data(train_ds,valid_ds,bs)
fit(steps,model,loss_func,optimizer,get_data)
'''
train_dl,valid_dl=get_data(train_ds,valid_ds,bs)
fit(steps,model,loss_func,optimizer,train_dl,valid_dl)
torch.save(save_model, 'last.pt')