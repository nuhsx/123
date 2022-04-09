import os 
import PIL
from PIL import Image
p_road='D:/vs project/dump_a_dump/photos/'
l_road='D:/vs project/dump_a_dump/labels/'
img_list=os.listdir(p_road) 
lab_list=os.listdir(l_road) 
x_train_path='D:/vs project/dump_a_dump/train3/Image/'
y_train_path='D:/vs project/dump_a_dump/train3/labels/'
x_valid_path='D:/vs project/dump_a_dump/val3/Image/'
y_valid_path='D:/vs project/dump_a_dump/val3/labels/'
import random
i_p=[]
j_l=[]
for i in img_list:
    i_p.append(i)
#print(i_p)
for j in lab_list:
    j_l.append(j)

for i in range(6289):

    r=random.randint(0,1)
    if r ==0:
        train=f'{x_train_path+i_p[i]}'
        tlabel=f'{y_train_path+j_l[i]}'
        im=Image.open(f'D:/vs project/dump_a_dump/photos/{i_p[i]}')
        im.save(train)
        tl=open(f'D:/vs project/dump_a_dump/labels/{j_l[i]}',encoding='utf-8')
        tl2=tl.read()
        file=open(tlabel, 'w')
        file.write(str(tl2))
        file.close()
    elif r==1:
        val=f'{x_valid_path+i_p[i]}'
        vlabel=f'{y_valid_path+j_l[i]}'
        im1=PIL.Image.open(f'D:/vs project/dump_a_dump/photos/{i_p[i]}')
        im1.save(val)
        vl=open(f'D:/vs project/dump_a_dump/labels/{j_l[i]}',encoding='utf-8')
        vl2=vl.read()
        file=open(vlabel, 'w')
        file.write(str(vl2))
        file.close()