#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


# In[2]:


x,y = make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x,y)


# In[3]:


print(x.shape)
y=y.reshape(y.shape[0],1)
print(y.shape)


# In[4]:


#matrice X
X=np.hstack((x,np.ones(x.shape)))
X.shape


# In[5]:


theta=np.random.randn(2,1)
theta


# In[6]:


def model(X, theta):
    return X.dot(theta)


# In[7]:


plt.scatter(x,y)
plt.plot(x, model(X, theta))


# In[8]:


def cost_function(X, y, theta):
    m=len(y)
    return 1/(2*m )* np.sum((model(X,theta)-y)**2)


# In[9]:


cost_function(X, y, theta)


# In[10]:


def grad(X,y, theta):
    m=len(y)
    return 1/m *X.T.dot(model(X,theta)-y)


# In[22]:


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history=np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta=theta-learning_rate*grad(X, y, theta)
        cost_history[i]=cost_function(X, y, theta)
    return theta, cost_history
    


# In[23]:


theta_final, cost_history= gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000)


# In[18]:


theta_final


# In[19]:


predictions= model(X, theta_final)
plt.scatter(x,y)
plt.plot(X, predictions, c='r')


# In[24]:


plt.plot(range(1000), cost_history)


# In[27]:


def coef_determination(y, pred):
    u=((y-pred)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-u/v


# In[28]:


coef_determination(y, predictions)


# In[29]:


from sklearn.linear_model import SGDRegressor


# In[31]:


np.random.seed(0)
x, y= make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x,y)
                      


# In[33]:


model=SGDRegressor(max_iter=100, eta0=0.0001)
model.fit(x,y)


# In[35]:


print('coeff R2=' , model.score(x,y))
plt.scatter(x,y)
plt.plot(x, model.predict(x) , c='red', lw=3)


# In[53]:


model=SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x,y)


# In[55]:


print('coeff R2=' , model.score(x,y))
plt.scatter(x,y)
plt.plot(x, model.predict(x) , c='red', lw=3)


# In[46]:


from sklearn.preprocessing import PolynomialFeatures


# In[47]:


np.random.seed(0)


# In[48]:


# création du Dataset 
x, y = make_regression(n_samples=100, n_features=1, noise=10) 
y = y**2 # y ne varie plus linéairement selon x !


# In[49]:


# On ajoute des variables polynômiales dans notre dataset
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x = poly_features.fit_transform(x)


# In[50]:


plt.scatter(x[:,0], y) 
x.shape # la dimension de x: 100 lignes et 2 colonnes


# In[51]:


# On entraine le modele comme avant ! rien ne change ! 
model = SGDRegressor(max_iter=1000, eta0=0.001) 
model.fit(x,y) 
print('Coeff R2 =', model.score(x, y))


# In[52]:


plt.scatter(x[:,0], y, marker='o') 
plt.scatter(x[:,0], model.predict(x), c='red', marker='+')


# In[ ]:




