import PIL
import os
from PIL import Image

#x_train_path='D:/vs project/dump_a_dump/train2/Image/'
box=(0,0,430,430)
img_list=os.listdir(x_train_path) 
for i in img_list:
    f =  (PIL.Image.open(f'{x_train_path}{i}').convert("L"))
    
    result=f.crop(box)
    #print(result.size)
    result.save(f'{x_train_path}{i}')
    #result.show()
'''
from PIL import ImageGrab
import cv2
import side

im = ImageGrab.grab()
im.save('current.png')
f =  cv2.imread('current.png')
    
center1=side.jump_peo(f)'''







