#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import numpy as np


# In[2]:


# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)  
X = sp.optimallhc(20)


# In[3]:


# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin  
y = testfun(X)


# In[9]:


X.shape


# In[11]:


X


# In[10]:


y.shape


# In[12]:


y


# In[14]:


# Now that we have our initial data, we can create an instance of a Kriging model
k = kriging(X, y, testfunction=testfun, name='simple')  
k.train()


# In[22]:


# Now, five infill points are added. Note that the model is re-trained after each point is added
numiter = 5  
for i in range(numiter):  
    print('Infill iteration {0} of {1}....'.format(i + 1, numiter))
    newpoints = k.infill(1)
    print(newpoints)
    for point in newpoints:
        k.addPoint(point, testfun(point)[0])        
    k.train()


# In[23]:


# And plot the results
k.plot()  


# In[38]:


# Predict some values for y for corresponding point in X
k.predict([0, 0])


# In[39]:


k.predict([1, 1])


# In[40]:


k.predict([1, 0])


# In[ ]:




