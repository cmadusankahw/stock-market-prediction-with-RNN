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


# In[3]:


from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

model=Sequential()

model.add(LSTM(units=96,return_sequences=True,input_shape=(data.shape[1:]))) 
#don't use many LSMT layers as they have a high memory capacity to be trained and no need of adding more
model.add(Dropout(0,2))
model.add(LSTM(units=96,return_sequences=True)) #units not to be same as previous layers, can be assumed differently
model.add(Dropout(0,2))
model.add(LSTM(units=96,return_sequences=True)) 
model.add(Dropout(0,2))
model.add(LSTM(units=96,return_sequences=False)) 
model.add(Dense(1)) #THIS IS A REGRESSION PROBLEM - LINEAR ACTIVATION (no need to mention as activation is by default linear)

model.compile(loss='mse',optimizer='adam') #loss is mean quared error for regression problems


# model.summary()

# In[4]:


model.summary()


# In[5]:


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


# In[18]:



plt.plot(test_target,'b',label='train_data')
plt.plot(result+0.015,'g',label='predicted_results')
plt.xlabel('Day')
plt.ylabel('Stock Value $')
plt.legend()
plt.show()


# In[20]:


from sklearn.metrics import r2_score

r2= r2_score(test_target,result)
print('R2 score',r2)


# In[ ]:




