#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

dataset=pd.read_csv('AAPL_data.csv').values

data=dataset[:,1]
print(data.shape)
print(data[:5])


# In[4]:


data=data.reshape(-1,1)
print(data.shape)


# In[5]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1)) #ccreate the scaling object -> values to between 0-1
data_scaled=scaler.fit_transform(data)


# In[6]:


data_window=[]
target=[]

for i in range (50,data.shape[0]):
    data_window.append(data_scaled[i-50:i,0]) #i=0 to i=50 (of first column)
    target.append(data_scaled[i,0])
    


# In[7]:


import numpy as np

data_np=np.array(data_window)
target_np = np.array(target)


# In[8]:


print (data_np.shape)


# In[9]:


data_reshaped=data_np.reshape(data_np.shape[0],data_np.shape[1],1) #shaped to 3d as the dataset is 1d


# In[10]:


print(data_reshaped.shape)


# In[ ]:


#can't use trainTestSplit as this is a time series problem, train test split will use random data. So we don't use it here


# In[12]:


np.save('data',data_reshaped)
np.save('target',target_np)


# In[ ]:




