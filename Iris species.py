#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.neural_network import MLPClassifier


# In[2]:


# charger les données 
iris = load_iris()


# In[3]:


X = iris.data 
y = iris.target


# In[4]:


X.shape # notre Dataset comprend 150 exemples et 4 variables


# In[5]:


# Visualisation des donées 
colormap=np.array(['Red','green','blue']) 
plt.scatter(X[:,3], X[:,1], c = colormap[y])


# In[6]:


# Création du modele 
model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000) 
model.fit(X, y) 
model.score(X, y)


# In[ ]:




