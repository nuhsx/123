import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.flatten import Flatten
class ConvBlock(nn.Module):
    """ 卷积块 """

    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              kernel_size=kernel_size, padding=padding)#7*7
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        #print(self.conv(x).shape)
        return F.relu(self.batch_norm(self.conv(x)))

class ValueHead(nn.Module):
    """ 价值头 """

    def __init__(self, in_channels, board_len):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        board_len: int
            棋盘大小
        """
        super().__init__()
        self.in_channels = in_channels
        self.board_len = board_len
        self.conv = ConvBlock(in_channels, 1, kernel_size=1)#降维
        self.fc = nn.Sequential(
            nn.Linear(board_len**2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x
class ResidueBlock(nn.Module):
    """ 残差块 """

    def __init__(self, in_channels=16   , out_channels=16):
        """
        Parameters
        ----------
        in_channels: int
            输入图像通道数

        out_channels: int
            输出图像通道数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out + x)

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight,0.01)
        nn.init.constant_(layer.bias, 0.1)

class nnet(nn.Module):
    def __init__(self,bs:int,inchannel=1):
        super().__init__()
        #权重初始化
        
        #self.shape=x.size[0]
        self.bs=bs
        self.conv1=nn.Conv2d(inchannel,16,5,2)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(16,32,5,2)
        self.conv3=nn.Conv2d(32,64,5,2)
        self.conv4= nn.Conv2d(64, 64, 5,2)
        self.conv5=nn.Conv2d(64,64,5,2)
        self.dropout=nn.Dropout(p=0.5)
        self.bn=nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn1=nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.residues = ResidueBlock()
        self.bn2=nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.linear1=nn.Linear(64,32)
        self.linear=nn.Linear(32,1)
    def forward(self,x):
        print(x.shape)
        
        #print(x.shape)
        
        out=self.conv1(x)
        out=self.bn1(out)
        #print(out)
        out=self.pool(out)
        out=F.relu(out)
        for i in range(3):
            out=self.residues(out)
            #print(out)
        
            
        out=F.relu(out)
        #out=self.pool(out)
        out=self.conv2(out)
        #print(out.shape)
        out=self.bn(out)
        
            #out=self.bn1(out)
        out=F.relu(out)
        out=self.pool(out)
        out=self.conv3(out)
        #print(out.shape)
        out=self.bn2(out)
        
            #out=self.bn1(out)
        out=F.relu(out)
        out=self.pool(out)
        #out=self.conv5(out)
        #print(out.shape)
        #out=self.bn(out)
        
            #out=self.bn1(out)
        out=F.relu(out)
        #out=self.pool(out)
        out=self.conv4(out)
        print(out.shape)
        #out=self.bn2(out)
        out=F.relu(out)
            #out=self.bn1(out)
        #out=F.relu(out)
        #out=self.pool(out)
        #out=out.flatten()
        #out=self.residues(out)
       
        #out=F.relu(out)
        
        #print(out.shape)
        #
   
        #out=self.pool(out)

        #out=self.bn(out)
        #out=self.pool(out)
        

        #out=out.reshape(self.bs,-1)
        
        out=self.dropout(out)
        out=out.view(-1,64)
        out=self.linear1(out)
        
        #out=nn.Linear(out.shape[1],32)(out)
        out=F.relu(out)
        
        #print(out.shape)
        #out=self.dropout(out)
        
        out=self.linear(out)
        out=F.relu(out)
        #print("out",out)
        #print(out)
        #out=nn.Sigmoid()(out)
        #out=nn.Softmax(dim=0)(out)#总和是1
        #inchannel,size=(out.shape)[1],(out.shape)[2]

        #print(out.shape)
        #out=self.pool(out)
        #print(out.shape)
        
        #out=self.conv3(out)
        #print(out.shape)
        #out=self.pool(out)
        #print(out.shape)
        #out=self.conv4(out)
        #print(out.shape)
        #out=ValueHead(inchannel,size)(out)
        #print(out.item())
        
        return out
#print(nnet(1)(torch.rand(1,1,430,430)))
'''
model=nnet(6).apply(init_weights)
print(list(model.parameters()))

print(nnet(6)(torch.rand(6,1,430,430)))
ran=torch.randn((6,1))
print(ran)
print(nn.Softmax(dim=0)(ran))
'''
'''
        #out=self.pool(out)
       
        out=out.reshape(1,-1)
        
        out=nn.Linear(out.shape[1],self.bs)(out)
        #print(out.shape)

        out=out.reshape(self.bs,1)
        #out=self.dropout(out)
        #out=F.relu(out)
        #out=self.dropout(out)
        
        #print(out.shape)
        out=nn.Sigmoid(dim=1)(out)
        
        print(out.shape)
        #out=F.relu(out)
        return out

print(nnet(4)(torch.rand(85,1,100,100)))
'''