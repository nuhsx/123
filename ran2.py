import os 
import PIL
from PIL import Image


x_valid_path='D:/vs project/dump_a_dump/val3/Image/'
y_valid_path='D:/vs project/dump_a_dump/val3/labels/'
img_list=os.listdir(x_valid_path) 
lab_list=os.listdir(y_valid_path) 
import random
i_p=[]
j_l=[]
for i in img_list:
    i_p.append(i)
#print(i_p)
for j in lab_list:
    j_l.append(j)
length=[]
x_train_path='D:/vs project/dump_a_dump/train3/Image/'
y_train_path='D:/vs project/dump_a_dump/train3/labels/'#fang
i=0
while i !=2309:
    
    r=random.randint(0,1)
    if r ==0:
        val=f'{x_valid_path+i_p[i]}'
        vlabel=f'{y_valid_path+j_l[i]}'#qu
        im1=PIL.Image.open(val)
        im1.save(f'{x_train_path+i_p[i]}')
        os.remove(val)
        vl=open(vlabel,encoding='utf-8')
        vl2=vl.read()
        file=open(f'{y_train_path+j_l[i]}', 'w')
        file.write(str(vl2))
        file.close()
        vl.close()
        os.remove(vlabel)
        i+=1