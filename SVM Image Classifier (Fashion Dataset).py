#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install --user tensorflow


# In[1]:


# Import Dependencies

import numpy as np # For Array processing 
from sklearn.svm import SVC # Classifier
import matplotlib.pyplot as plt # For plotting images
import random # Randomly viewing images
import time # For calculating run times


# In[2]:


start = time.time()
from keras.datasets import fashion_mnist # Import Dataset from Keras Package

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data() # Load train and Test Datasets
print(str(time.time()-start)+" Seconds to load the data")


# In[6]:


# Randomly sample Images

# Defining Label Names
labels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(15, 15)) 
for i in range(10 * 10):
    rand = random.randint(0, len(train_X)+1)
    image = train_X[rand] 
    plt.subplot(10, 10, i+1)       
    plt.imshow(image)  
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(labels[train_Y[rand]])
    plt.tight_layout()   
    
plt.show()


# In[14]:


# Image Preprocessing

#Reshape image

train_x = train_X.reshape(train_X.shape[0],train_X.shape[1]*train_X.shape[2])
test_x = test_X.reshape(test_X.shape[0],test_X.shape[1]*test_X.shape[2])
train_x=train_x.astype('float32')
test_x=test_x.astype('float32')

train_x /= 255.0
test_x /= 255.0

print(train_x.shape)
print(test_x.shape)


# In[16]:


# SVM Model
start1 = time.time()

svc = SVC(C=1, kernel='linear', gamma="auto")
svc.fit(train_x, train_Y)

end1 = time.time()
svm_time = end1-start1
print(svm_time)


# In[ ]:


# Save the model as a pickle object
file = "svm_model.sav"

import pickle

dump(svc, open(file, 'wb'))

