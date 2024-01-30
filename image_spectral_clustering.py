#!/usr/bin/env python
# coding: utf-8

# In[38]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


# # Generate Data Points

# <img src="data/moon_shapes.png" width='250'>

# In[41]:


raw_moon_data = datasets.make_moons(n_samples=1000)
len(raw_moon_data)


# In[42]:


moon_data = raw_moon_data[0]
print(moon_data.shape)


# In[43]:


moon_labels = raw_moon_data[1]
print(moon_labels.shape)


# In[44]:


# visualize it
plt.plot(moon_data[:,0], moon_data[:,1], 'o')


# In[47]:


moon_data_min = np.min(moon_data)
moon_data_max = np.max(moon_data)
moon_data_mean = np.mean(moon_data)
moon_data_std = np.std(moon_data)
print(f'min: {moon_data_min}')
print(f'max: {moon_data_max}')
print(f'mean: {moon_data_mean}')
print(f'std: {moon_data_std}')


# In[49]:


# create noise
noises = np.random.normal(loc = moon_data_mean, # mean
                          scale = moon_data_std, # std
                          size = (data.shape[0], data.shape[1]))

noises


# In[50]:


moon_data_jitter = moon_data + noises

moon_data_j_min = np.min(moon_data_jitter)
moon_data_j_max = np.max(moon_data_jitter)
moon_data_j_mean = np.mean(moon_data_jitter)
moon_data_j_std = np.std(moon_data_jitter)
print(f'min: {moon_data_j_min}')
print(f'max: {moon_data_j_max}')
print(f'mean: {moon_data_j_mean}')
print(f'std: {moon_data_j_std}')


# In[51]:


# visualize it
plt.plot(moon_data_jitter[:,0], moon_data_jitter[:,1], 'o')


# It was too noisy and entirely messed up the moon shapes.

# In[55]:


# create noise#2
noises = np.random.normal(loc = 0, # mean
                          scale = 0.1, # std
                          size = (data.shape[0], data.shape[1]))

moon_data_jitter = moon_data + noises
plt.plot(moon_data_jitter[:,0], moon_data_jitter[:,1], 'o')


# Harded coded standard deviation worked much better.

# So `moon_data_jitter` will be my input for the spectral clustering.

# <img src="data/two_rings.png" width='250'>

# In[59]:


angle = np.random.randint(low = 0, high = 359, size = 1000)


# In[60]:


center = np.array([0, 0])


# In[ ]:





# # Spectral Clustering

# In[56]:


moon_data_jitter


# In[ ]:




