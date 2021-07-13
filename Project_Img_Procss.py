import numpy as np
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

# prior processing the images:
#transform = transforms.Compose([transforms.Resize((600, 600)), transforms.Grayscale(1), transforms.ToTensor()]) #working good!!!
transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
root = "./archive_chest_scan_imgs"

# load image with labels - which image came from each folders
dataset = datasets.ImageFolder(root=root, transform=transform)
#print(len(dataset)) #get dataset length

#Get max and min resolution of the images
"""
max_y_axis = 0
min_y_axis = 2000
for image in dataset:
    if len(image[0][0][:]) > max_y_axis:
        max_y_axis = len(image[0][0][:])
    if len(image[0][0][:]) < min_y_axis:
        min_y_axis = len(image[0][0][:])

print("max_y_axis: ",max_y_axis)
print("min_y_axis: ",min_y_axis)

max_x_axis = 0
min_x_axis = 2000
for image in dataset:
    if len(image[0][0][0]) > max_x_axis:
        max_x_axis = len(image[0][0][0])
    if len(image[0][0][0]) < min_x_axis:
        min_x_axis = len(image[0][0][0])

print("max_x_axis: ",max_x_axis)
print("min_x_axis: ",min_x_axis)

print("avg y axis: ",(int(max_y_axis)+int(min_y_axis))/2)
print("avg x axis: ",(int(max_x_axis)+int(min_x_axis))/2)
"""

# plot one image for example:
img = dataset[0][0]
plt.imshow(img.permute(1, 2, 0))
#print(img.shape) #get image dimensions

# print(len(dataset[0][0][0][:])) #get image y axis
# print(len(dataset[0][0][0][0])) #get image x axis


#Augmentation part code:
def visualize(original, augmented):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)

from PIL import Image
img = Image.open("C:/Users/kfiri/PycharmProjects/ImageProcessingBigProject/archive_chest_scan_imgs/Data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000005 (3).png")
flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

visualize(img, flipped)
plt.show()




"""
#mirror all data:
import tensorflow_datasets as tfds

def augment(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
  image = tf.image.random_brightness(image, max_delta=0.5)
  return image, label

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
     with_info=True,
     as_supervised=True,)

train_ds = train_ds.shuffle(1000).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
"""

