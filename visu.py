
# coding: utf-8

# In[1]:



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import glob
from PIL import Image
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import torch.autograd as ag
from sklearn.model_selection import train_test_split


# In[2]:


random.seed( 42 )


# In[24]:


base_dir = "/datasets/ee285f-public/human-protein"
train_image_dir = os.path.join(base_dir, 'train')
test_image_dir = os.path.join(base_dir, 'test')


# In[25]:


df_train = pd.read_csv(base_dir + '/train.csv')
df_test = pd.DataFrame()


# In[26]:


df_train['path'] = df_train['Id'].map(lambda x: os.path.join(train_image_dir, '{}_green.png'.format(x)))
df_train['target_list'] = df_train['Target'].map(lambda x: [int(a) for a in x.split(' ')])

df_test['path'] = glob.glob(os.path.join(test_image_dir, '*.png'))


# In[27]:


df_train.head()


# In[28]:


X = df_train['path'].values
print(X.shape)
y = df_train['target_list'].values

X_test = df_test['path'].values


# In[29]:


class CellsDataset(Dataset):

    def __init__(self, X, y=None, transforms=None, nb_organelle=28):
        
        self.nb_organelle = nb_organelle
        self.transform = transforms 
        self.X = X
        self.y = y
            
    def open_rgby(self, path2data): #a function that reads RGBY image
        
        Id = path2data.split('/')[-1].split('_')[0]
        basedir = '/'.join(path2data.split('/')[:-1])
        
        images = np.zeros(shape=(512,512,3))
        colors = ['red','green','blue']
        for i, c in enumerate(colors):
            images[:,:,i] = np.asarray(Image.open(basedir + '/' + Id + '_' + c + ".png"))
        
            yellow_ch = np.asarray(Image.open(basedir + '/' + Id + '_yellow.png'))
            images[:,:,0] += (yellow_ch/2).astype(np.uint8) 
            images[:,:,1] += (yellow_ch/2).astype(np.uint8)
            #print(images.shape)
        
        return images.astype(np.uint8)
    
    def __getitem__(self, index):
        
        path2img = self.X[index]
        image = self.open_rgby(path2img)

        if self.y is None:
            labels =np.zeros(self.nb_organelle,dtype=np.int)
        else:
            label = np.eye(self.nb_organelle,dtype=np.float)[self.y[index]].sum(axis=0)
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.X)


# In[30]:


class AdjustGamma(object):
    def __call__(self, img):
        return transforms.functional.adjust_gamma(img, 0.8, gain=1)


# In[31]:


class AdjustContrast(object):
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, 2)


# In[32]:


class AdjustBrightness(object):
    def __call__(self, img):
        return transforms.functional.adjust_brightness(img, 2)


# In[33]:


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

def denormalize(image, mean=imagenet_mean, std=imagenet_std):
    inp = image.transpose((1, 2, 0))
    img = std * inp + mean
    return img


# In[34]:


data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(), # because the input dtype is numpy.ndarray
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AdjustGamma(),
        AdjustContrast(),
        ##AdjustBrightness(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]),
}


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)


# In[36]:


X_train, X_valid, y_train, y_valid = train_test_split(
     X_train, y_train, test_size=0.25, random_state=42)


# In[37]:


dsets = {
    'train': CellsDataset(X_train, y_train, transforms=data_transforms['train']),
    'valid': CellsDataset(X_valid, y_valid, transforms=data_transforms['test']),
    'test':  CellsDataset(X_test, y_test,  transforms=data_transforms['test']),
}


# In[38]:


batch_size = 32
random_seed = 3
valid_size = 0.4
shuffle = True


# In[39]:


def create_dataLoader(dsets, batch_size, shuffle=False, pin_memory=False):
    
    dset_loaders = {} 
    for key in dsets.keys():
        if key == 'test':
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle=False)
        else:
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    return dset_loaders


# In[40]:


dset_loaders = create_dataLoader(dsets, batch_size, shuffle, pin_memory=False)
#returning this dset_loaders provides the data in minibatches of 32.


# In[41]:


dset_loaders.keys()



def plot_organelles(dset_loaders, is_train = True, preds_test = [], preds_train = []):
    X, y = next(iter(dset_loaders))
    X, y = X.numpy(), y.numpy()
    plt.figure(figsize=(20,10))
    for i in range(0, 4):
        plt.subplot(1,4,i+1)
        rand_img = random.randrange(0, X.shape[0])
        img = denormalize(X[rand_img,:,:,:])
        img = np.clip(img, 0, 1.0)    
        plt.imshow(img)
        plt.axis('off')


# In[3]:


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

def denormalize(image, mean=imagenet_mean, std=imagenet_std):
    inp = image.transpose((1, 2, 0))
    img = std * inp + mean
    return img


# In[2]:


plot_organelles(dset_loaders['train'])

