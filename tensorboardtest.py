from torch.utils.tensorboard import SummaryWriter
writer= SummaryWriter("l")#命令行输入tensorboard --logdir=logs
for i in range (100):
    writer.add_scalar("y=x",i,i)


writer.close()