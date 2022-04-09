'''
from numpy.lib.function_base import average, copy
import torch
from torch.nn.parameter import Parameter
from torch import nn,optim
from torch.functional import Tensor
from torch.nn import functional as F
import torchvision
class ConvBlock(nn.Module):
    """ 卷积块 """

    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              kernel_size=kernel_size, padding=padding)#7*7
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        print(self.conv(x).shape)
        return F.relu(self.batch_norm(self.conv(x)))

class PolicyHead(nn.Module):
    """ 策略头 """

    def __init__(self):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        board_len: int
            棋盘大小
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels=64,out_channels=2, kernel_size=1, padding=0)#改【1，2，25，25】
        
        self.batch_norm = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2*81, 81)#输出 board_len**2个值

    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        
        x = self.fc(x.flatten(1))
        return F.log_softmax(x, dim=1)
class ResidueBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(64,64,3,1,1)
        self.conv2=nn.Conv2d(64,64,3,1,1)
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(64)
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        return out+x

class ValueHead(nn.Module):
    """ 价值头 """

    def __init__(self, in_channels=64, board_len=9):
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
        self.conv = ConvBlock(64, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(870*470, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            #nn.ReLU()
            nn.Tanh()
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x

class Net(nn.Module):
    

    def __init__(self):#参数要改
        super().__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1)#??? 1,64,109,109
        #self.conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)#输入的通道数=输出，不改变
        self.R=ResidueBlock()
        self.time_head = ValueHead(128,9)
       
        #self.valuehead=ValueHead()
        #self.policy_head = PolicyHead()#棋盘的预测（列表）
        #self.relu1=F.relu()relu不需要参数

    def forward(self,x):
        x=x.reshape(-1,1,870,470)

        print(x.shape)
        out =self.conv1(x)#1t
        #print(out.shape)
        print(out.shape)
        out=self.bn1(out)
        print(out.shape)
        out=F.relu(out)
        print(out.shape)
        out=self.R(out)
        print(out.shape)
        out=self.R(out)
        print(out.shape)
        out=self.R(out)
        print(out.shape)
        #out=self.R(out)
        #print(out.shape)
        out=self.time_head(out)
        

        
        print(out.shape)
        
        
        
        
        
        #board_p=self.policy_head(out)
        #board_v=self.valuehead(out)
        #out=self.bn2(out)
        #out= F.relu(out)        
        return out
#a=Net()(torch.randn(1,1,9,9))[0][0]
#print(Net()(torch.rand(8,3,870,470)))
    '''