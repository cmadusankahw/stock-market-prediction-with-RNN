#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')


# In[2]:


#manually idenifing train test splits
train_data=data[:1000]
train_target=target[:1000]

test_data=data[1000:]
test_target=target[1000:]


# In[10]:


from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,Dropout,Activation,Flatten #1D because time series data

model=Sequential()

model.add(Conv1D(filters=128,input_shape=(data.shape[1:]),kernel_size=5))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4))

model.add(Conv1D(filters=64,input_shape=(data.shape[1:]),kernel_size=5))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

model.summary()


# In[11]:


model.fit(train_data,train_target,epochs=100)


# In[13]:


result=model.predict(test_data)


# In[14]:


from matplotlib import pyplot as plt

plt.plot(train_target,'r',label='train_data')
plt.plot(np.arange(1000,1209),test_target,'b',label='train_data')
plt.plot(np.arange(1000,1209),result,'g',label='predicted_results')
plt.xlabel('Day')
plt.ylabel('Stock Value $')
plt.legend()
plt.show()


# In[16]:



plt.plot(test_target,'b',label='train_data')
plt.plot(result,'g',label='predicted_results')
plt.xlabel('Day')
plt.ylabel('Stock Value $')
plt.legend()
plt.show()


# In[17]:


from sklearn.metrics import r2_score

r2= r2_score(test_target,result)
print('R2 score',r2)


# In[ ]:




