#!/usr/bin/env python
# coding: utf-8

# # Project
# 
# ## data preprocessing and shuffling code

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import pathos.multiprocessing as pmp # -pip install pathos (if not on machine)

print("TensorFlow version: " + tf.__version__)


# In[2]:


import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
import tensorflow.keras

print('Loading Adience dataset...')

fold_id=0

#from read_processed_data import fetch_data_aligned # old data
from read_processed_data_crop import fetch_data_cropped


# In[3]:


tf.keras.backend.clear_session()


# ### Fold1

# In[4]:


age, gender, RGB_data = fetch_data_cropped(0)

num_images = RGB_data.shape[0] # put it 26580 (all images in the 5 folders)
img_dim = RGB_data.shape[1]     # put it 816
num_ch = RGB_data.shape[3]      # 3
age_classes = 8 # from 0 to 7
gender_classes = 2 # 0/1 i.e., m/f

# Creating the 3 data arrays
rgb_data = np.array(RGB_data[:,:,:,:]) #np.random.rand(num_images, img_dim*img_dim*num_ch) # array size = 26580x1997568
age = np.array(age[:].reshape(age.shape[0], 1)) #np.random.randint(age_classes, size=(num_images,1)) # array size = 26580x1
gender = np.array(gender[:].reshape(gender.shape[0],1)) #np.random.randint(2, size=(num_images,1)) # array size = 26580x1


# In[5]:


y = np.array(range(len(RGB_data[:,:,:,:])))
y = shuffle(y, random_state=0)

rgb_data = rgb_data[y,:,:,:]
age = age[y]
gender = gender[y]


# ### Fold2

# In[6]:


age2, gender2, RGB_data2 = fetch_data_cropped(1)

num_images = RGB_data2.shape[0] # put it 26580 (all images in the 5 folders)
img_dim = RGB_data2.shape[1] # put it 816
num_ch = RGB_data2.shape[3] # 3
age_classes = 8 # from 0 to 7
gender_classes = 2 # 0/1 i.e., m/f

# Creating the 3 data arrays
rgb_data2 = np.array(RGB_data2[:,:,:,:]) #np.random.rand(num_images, img_dim*img_dim*num_ch) # array size = 26580x1997568
age2 = np.array(age2[:].reshape(age2.shape[0], 1)) #np.random.randint(age_classes, size=(num_images,1)) # array size = 26580x1
gender2 = np.array(gender2[:].reshape(gender2.shape[0],1)) #np.random.randint(2, size=(num_images,1)) # array size = 26580x1


# In[7]:


y = np.array(range(len(RGB_data2[:,:,:,:])))
y = shuffle(y, random_state=0)

rgb_data2 = rgb_data2[y,:,:,:]
age2 = age2[y]
gender2 = gender2[y]


# ### Fold3

# In[8]:


age3, gender3, RGB_data3 = fetch_data_cropped(2)

num_images = RGB_data3.shape[0] # put it 26580 (all images in the 5 folders)
img_dim = RGB_data3.shape[1] # put it 816
num_ch = RGB_data3.shape[3] # 3
age_classes = 8 # from 0 to 7
gender_classes = 2 # 0/1 i.e., m/f

# Creating the 3 data arrays
rgb_data3 = np.array(RGB_data3[:,:,:,:]) #np.random.rand(num_images, img_dim*img_dim*num_ch) # array size = 26580x1997568
age3 = np.array(age3[:].reshape(age3.shape[0], 1)) #np.random.randint(age_classes, size=(num_images,1)) # array size = 26580x1
gender3 = np.array(gender3[:].reshape(gender3.shape[0],1)) #np.random.randint(2, size=(num_images,1)) # array size = 26580x1


# In[9]:


y = np.array(range(len(RGB_data3[:,:,:,:])))
y = shuffle(y, random_state=0)

rgb_data3 = rgb_data3[y,:,:,:]
age3 = age3[y]
gender3 = gender3[y]


# ### Fold4

# In[10]:


age4, gender4, RGB_data4 = fetch_data_cropped(3)

num_images = RGB_data4.shape[0] # put it 26580 (all images in the 5 folders)
img_dim = RGB_data4.shape[1] # put it 816
num_ch = RGB_data4.shape[3] # 3
age_classes = 8 # from 0 to 7
gender_classes = 2 # 0/1 i.e., m/f

# Creating the 3 data arrays
rgb_data4 = np.array(RGB_data4[:,:,:,:]) #np.random.rand(num_images, img_dim*img_dim*num_ch) # array size = 26580x1997568
age4 = np.array(age4[:].reshape(age4.shape[0], 1)) #np.random.randint(age_classes, size=(num_images,1)) # array size = 26580x1
gender4 = np.array(gender4[:].reshape(gender4.shape[0],1)) #np.random.randint(2, size=(num_images,1)) # array size = 26580x1


# In[11]:


y = np.array(range(len(RGB_data4[:,:,:,:])))
y = shuffle(y, random_state=0)

rgb_data4 = rgb_data4[y,:,:,:]
age4 = age4[y]
gender4 = gender4[y]


# ### Fold5 The last one

# In[12]:


age5, gender5, RGB_data5 = fetch_data_cropped(4)

num_images = RGB_data5.shape[0] # put it 26580 (all images in the 5 folders)
img_dim = RGB_data5.shape[1] # put it 816
num_ch = RGB_data5.shape[3] # 3
age_classes = 8 # from 0 to 7
gender_classes = 2 # 0/1 i.e., m/f

# Creating the 3 data arrays
rgb_data5 = np.array(RGB_data5[:,:,:,:]) #np.random.rand(num_images, img_dim*img_dim*num_ch) # array size = 26580x1997568
age5 = np.array(age5[:].reshape(age5.shape[0], 1)) #np.random.randint(age_classes, size=(num_images,1)) # array size = 26580x1
gender5 = np.array(gender5[:].reshape(gender5.shape[0],1)) #np.random.randint(2, size=(num_images,1)) # array size = 26580x1


# In[13]:


y = np.array(range(len(RGB_data5[:,:,:,:])))
y = shuffle(y, random_state=0)

rgb_data5 = rgb_data5[y,:,:,:]
age5 = age5[y]
gender5 = gender5[y]


# ### Concatting and data is shuffled

# In[14]:


rgb_data = np.concatenate((rgb_data,rgb_data2), axis=0)
del rgb_data2
rgb_data = np.concatenate((rgb_data,rgb_data3), axis=0)
del rgb_data3
rgb_data = np.concatenate((rgb_data,rgb_data4), axis=0)
del rgb_data4
rgb_data = np.concatenate((rgb_data,rgb_data5), axis=0)
del rgb_data5


# In[15]:


age = np.concatenate((age,age2), axis=0)
del age2
age = np.concatenate((age,age3), axis=0)
del age3
age = np.concatenate((age,age4), axis=0)
del age4
age = np.concatenate((age,age5), axis=0)
del age5


# In[16]:


gender = np.concatenate((gender,gender2), axis=0)
del gender2
gender = np.concatenate((gender,gender3), axis=0)
del gender3
gender = np.concatenate((gender,gender4), axis=0)
del gender4
gender = np.concatenate((gender,gender5), axis=0)
del gender5


# ### Encoding shuffled output classes - age and gender

# In[17]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(age)
age_enc=le.transform(age)
del age


# In[18]:


le = preprocessing.LabelEncoder()
le.fit(gender)
gender_enc=le.transform(gender)
del gender


# ### Folds - 10 folds

# In[19]:


'Step 5:'
# Getting the 10-fold training, validation and testing data sets
kf = KFold(n_splits=10, shuffle=False, random_state=None)
kf.get_n_splits(rgb_data)

for train_index, test_index in kf.split(rgb_data):
    print("TRAIN-VALID:\n", train_index, "\nTEST:", test_index)
    rgb_data_train, rgb_data_test = rgb_data[train_index], rgb_data[test_index]
    age_enc_train, age_enc_test = age_enc[train_index], age_enc[test_index]
    gender_train, gender_test = gender_enc[train_index], gender_enc[test_index]


# In[20]:


del rgb_data
del age_enc
del gender_enc


# ### Normalizing data to grayscale

# In[25]:


rgb_data_train = rgb_data_train.astype('float32')/255
rgb_data_test = rgb_data_test.astype('float32')/255

age_enc_train = age_enc_train.astype('float32')/255
age_enc_test = age_enc_test.astype('float32')/255

gender_train = gender_train.astype('float32')/255
gender_test = gender_test.astype('float32')/255


# In[ ]:




