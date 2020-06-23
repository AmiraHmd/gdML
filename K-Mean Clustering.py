#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


# In[8]:


# Générer des données:
X, y = make_blobs(n_samples=100, centers = 3, cluster_std=0.5, random_state=0) #nb_feat
ures =2 
plt.scatter(X[:,0], X[:, 1])


# In[9]:


# Entrainer le modele de K-mean Clustering
model = KMeans(n_clusters=3)
model.fit(X)


# In[10]:


#Visualiser les Clusters
predictions = model.predict(X)
plt.scatter(X[:,0], X[:,1], c=predictions)


# In[12]:


# Entrainer le modele de K-mean Clustering
model = KMeans(n_clusters=2)
model.fit(X)


# In[13]:


#Visualiser les Clusters
predictions = model.predict(X)
plt.scatter(X[:,0], X[:,1], c=predictions)


# In[14]:


# Entrainer le modele de K-mean Clustering
model = KMeans(n_clusters=4)
model.fit(X)


# In[15]:


#Visualiser les Clusters
predictions = model.predict(X)
plt.scatter(X[:,0], X[:,1], c=predictions)


# In[ ]:




