#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split


# In[2]:


# Creation d'un Dataset Aleatoire 
np.random.seed(0) 
x, y = make_regression(n_samples=100, n_features=1, noise=10) 
y = np.abs(y) + y + np.random.normal(-5, 5, 100) 
plt.scatter(x, y)


# In[3]:


# Creation des Train set et Test set a partir du Dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[4]:


# Visualisation des Train set et Test set 
plt.scatter(x_train, y_train, c='blue', label='Train set') 
plt.scatter(x_test, y_test, c='red', label='Test set') 
plt.legend()


# In[5]:


X = PolynomialFeatures(degree = 10, include_bias=False).fit_transform(x)


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[7]:


from sklearn.linear_model import Ridge


# In[8]:


model = Ridge(alpha = 0.1, random_state=0) 
model.fit(x_train, y_train)


# In[9]:


print('Coefficient R2 sur Train set:', model.score(x_train, y_train))
print('Coefficient R2 sur Test set:', model.score(x_test, y_test))


# In[10]:


plt.figure(figsize=(8,6)) 
plt.scatter(x, y, c='blue') 
a = np.linspace(-2, 2, 100).reshape((100, 1)) 
A = PolynomialFeatures(degree = 10, include_bias=False).fit_transform(a) 
plt.plot(a, model.predict(A), c = 'green', lw=2)


# In[ ]:




