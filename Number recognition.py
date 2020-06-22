#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# importons une base de données de chiffre 
digits = load_digits()


# In[3]:


X = digits.data 
y = digits.target


# In[4]:


print('dimension de X:', X.shape)


# In[5]:


# visualisons un de ces chiffres
plt.imshow(digits['images'][0], cmap = 'Greys_r')


# In[6]:


# Entraînement du modele 
model = KNeighborsClassifier()
model.fit(X, y) 
model.score(X, y)


# In[7]:


#Test du modele 
test = digits['images'][100].reshape(1, -1) 
plt.imshow(digits['images'][100], cmap = 'Greys_r') 
model.predict(test)


# In[ ]:




