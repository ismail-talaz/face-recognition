import os
import cv2
import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import albumentations as alb

augmentornormal = alb.Compose([alb.HorizontalFlip(p=1.0), 
                         alb.RandomBrightnessContrast(p=0.5),
                         alb.RandomGamma(p=0.5), 
                         alb.RGBShift(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))

def set_photos(dset="train", index=2, target=1000, augmentor=augmentornormal, serial=1):

    file_path = str(os.path.join('celebA','Anno','list_bbox_celeba.txt'))

    with open(file_path, 'r') as log:
        
        i = 0
        for line in log:
            if(i < index):
                i+=1
                continue

            if(i == target):
                break
            
            
            splitted = line.split()
            image_path = splitted[0]

            coords = list(map(int,splitted[1:]))
            coords[2] = coords[2]+coords[0]
            coords[3] = coords[1]+coords[3]
            if(coords[3]*0.9 > coords[1]): coords[3] *= 0.90

            img = cv2.imread(os.path.join('celebA','Img','images','img_celeba',image_path))
            w, h = img.shape[1],img.shape[0]



            if ((((coords[3]-coords[1])*(coords[2]-coords[0]))/(h*w))<0.08 and serial==1):
                left = (img[:,int(coords[0]):]).copy()
                cv2.imwrite(os.path.join('aug_data', dset, 'images', f'left{image_path.split(".")[0]}.jpg'), left)
                with open(os.path.join('aug_data', dset, 'labels', f'left{image_path.split(".")[0]}.json'), 'w') as f:
                    json.dump({"class":1,"bbox":[0,coords[1]/left.shape[0],(coords[2]-coords[0])/left.shape[1],coords[3]/left.shape[0]]}, f)

                right = (img[:,:int(coords[2])]).copy()
                cv2.imwrite(os.path.join('aug_data', dset, 'images', f'right{image_path.split(".")[0]}.jpg'), right)
                with open(os.path.join('aug_data', dset, 'labels', f'right{image_path.split(".")[0]}.json'), 'w') as f:
                    json.dump({"class":1,"bbox":[coords[0]/right.shape[1],coords[1]/right.shape[0],coords[2]/right.shape[1],coords[3]/right.shape[0]]}, f)

                up = (img[int(coords[1]):,:]).copy()
                cv2.imwrite(os.path.join('aug_data', dset, 'images', f'up{image_path.split(".")[0]}.jpg'), up)
                with open(os.path.join('aug_data', dset, 'labels', f'up{image_path.split(".")[0]}.json'), 'w') as f:
                    json.dump({"class":1,"bbox":[coords[0]/up.shape[1],0,coords[2]/up.shape[1],(coords[3]-coords[1])/up.shape[0]]}, f)

                bottom = (img[:int(coords[3]),:]).copy()
                cv2.imwrite(os.path.join('aug_data', dset, 'images', f'bottom{image_path.split(".")[0]}.jpg'), bottom)
                with open(os.path.join('aug_data', dset, 'labels', f'bottom{image_path.split(".")[0]}.json'), 'w') as f:
                    json.dump({"class":1,"bbox":[coords[0]/bottom.shape[1],coords[1]/bottom.shape[0],coords[2]/bottom.shape[1],coords[3]/bottom.shape[0]]}, f)

            coords = list(np.divide(coords, [w,h,w,h]))
            
            cropper = alb.Compose([alb.RandomCrop(width=int(w*(2/3)), height=int(h*(2/3))), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
            
          
            augmented = cropper(image=img, bboxes=[coords], class_labels=['face']) if serial==1 else augmentornormal(image=img, bboxes=[coords], class_labels=['face'])
            cv2.imwrite(os.path.join('aug_data', dset, 'images', f'{image_path.split(".")[0]}-{serial}.jpg'), augmented['image'])

            annotation = {}

            if len(augmented['bboxes']) == 0: 
                annotation['bbox'] = [0,0,0,0]
                annotation['class'] = 0 

            else: 
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 1

            with open(os.path.join('aug_data', dset, 'labels', f'{image_path.split(".")[0]}-{serial}.json'), 'w') as f:
                json.dump(annotation, f)


            i+=1

def execute(total, train_split, val_split, test_split):          # Main Function to execute data augmentation
    checkpoint1 = total*train_split
    checkpoint2 = checkpoint1 + total*val_split
    checkpoint3 = checkpoint2 + total*test_split
    
    set_photos("train",2,checkpoint1,serial=1)
    set_photos("val",checkpoint1,checkpoint2,serial=1)
    set_photos("test",checkpoint2,checkpoint3,serial=1) 
    set_photos("train",2,checkpoint1,serial=2)
    set_photos("val",checkpoint1,checkpoint2,serial=2)
    set_photos("test",checkpoint2,checkpoint3,serial=2)


