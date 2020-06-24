#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression


# In[2]:


# Creation d'un Dataset x, y 
np.random.seed(0) 
x = np.linspace(0, 5, 10)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-2, 2, 10)
plt.scatter(x, y)


# In[3]:


# Creation de plusieurs features pour notre modele
X = x[:, np.newaxis] 
X = PolynomialFeatures(degree=10, include_bias=False).fit_transform(X) 
X.shape


# In[4]:


# Entraînement du modele. Ici on utilise les Equations Normales (LinearRegression)
# Les Equations normales reposent sur la méthode des moindres carrées, c’est plus 
# rapide que le Gradient Descent.
model = LinearRegression() 
model.fit(X,y) 
print('Coeff R2 =', model.score(X, y)) 
plt.scatter(x, y, marker='o') 
plt.plot(x, model.predict(X), c='red')


# In[6]:


from sklearn.linear_model import Ridge


# In[7]:


ridge = Ridge(alpha=0.1) # alpha est le facteur de régularisation. 
ridge.fit(X,y) 
print('Coeff R2 =', ridge.score(X, y))


# In[8]:


plt.scatter(x, y, marker='o') 
plt.plot(x, ridge.predict(X), c = 'green')


# In[ ]:




