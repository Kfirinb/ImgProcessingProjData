import pathlib
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from torchvision import datasets
import math
import sklearn.manifold as s
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from math import floor, ceil
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import torch.utils
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import applications

torch.cuda.is_available()

#AUGMENTATION:
root = "./archive_chest_scan_imgs/Data"
datagen = ImageDataGenerator(
        rotation_range=5,     #Random rotation between 0 and =chosen num
        horizontal_flip=False,
        fill_mode='reflect', cval=125)

train_valid_test = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
print(train_valid_test)
for m in range(len(train_valid_test)):
    sub_path = './archive_chest_scan_imgs/Data/' + train_valid_test[m]


    sub_folders = [name for name in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, name))]

    for i in range(len(sub_folders)):
        full_path = sub_path+'/'+sub_folders[i]
        full_path_original = sub_path+'/'+sub_folders[i] + '/' + "ORIGINAL_" + sub_folders[i]
        curr_len = len([name for name in os.listdir(full_path_original) if os.path.isfile(os.path.join(full_path_original,name))])
        print(curr_len)
        j = 0
        save_dir = full_path + '/' + "AUGMENTED_" + sub_folders[i]
        for batch in datagen.flow_from_directory(directory=full_path,
                                                 batch_size=1,
                                                 target_size=(256, 256),
                                                 color_mode="grayscale",
                                                 save_to_dir=save_dir,
                                                 save_prefix='aug',
                                                 save_format='png',
                                                 class_mode = 'categorical'):
            j += 1
            if j >= curr_len:
                break




#Model:



###~~~~~~~~~~~~~~~~~~~###
'''
# lets look at the filepath
root = "./archive_chest_scan_imgs/Data/train"
filedirectory = []
for files in os.listdir(root):
     filedirectory.append(os.path.join(root,files))
#filedirectory





###~~~~~~~~~~~~~~~~~~~###
# File Directory for both the train and test
train_path = "./archive_chest_scan_imgs/Data/train"
test_path = "./archive_chest_scan_imgs/Data/test"
# given that the dataset is really small, data augementation will be a good option to improve the accruracy
train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                  horizontal_flip = True,
                                  fill_mode = 'nearest',
                                  zoom_range=0.2,
                                  shear_range = 0.2,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  rotation_range=0.4)
train_generator = train_datagen.flow_from_directory(train_path,
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_path,
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')
'''




# prior processing the images:
root = "./archive_chest_scan_imgs"
# transform = transforms.Compose([transforms.Resize((600, 600)), transforms.Grayscale(1), transforms.ToTensor()]) #working good!!!
transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
# load image with labels - which image came from each folders
dataset = datasets.ImageFolder(root=root, transform=transform)
