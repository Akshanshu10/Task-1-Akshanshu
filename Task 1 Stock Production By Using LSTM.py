#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Task1 : Stock Market Prediction And Forecasting Using Stacked LSTM
Name - Akshanshu Bharti
What is LSTM ?
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture,
[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. 
It can process not only single data points (such as images), but also entire sequences of data (such as speech or video). 
For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition,
[2] speech recognition[3][4] and anomaly detection in network traffic or IDSs (intrusion detection systems).


# In[8]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


data.tail()


# In[12]:


plt.figure(figsize = (10,4))
plt.title('Tata Stocks Closing Price', color='Blue')
plt.plot(data['Close'], color='Black')
plt.xlabel('Date',fontsize=20, color='Blue')
plt.ylabel('Close',fontsize=20, color='Blue')


# In[13]:


data_close = data.reset_index()['Close']
data_close.head()


# In[14]:


from sklearn.preprocessing import MinMaxScaler 
import math 
import seaborn as sb 
from sklearn.metrics import mean_squared_error


# In[15]:


scaler = MinMaxScaler(feature_range = (0, 1))
data_close = scaler.fit_transform(np.array(data_close).reshape(-1, 1))
data_close


# In[17]:


#Training data 85 % and Testing data 15%
train_size = int(len(data_close) * 0.85)
test_size = len(data_close) - train_size
train_data, test_data = data_close[0 : train_size, :], data_close[train_size : len(data_close), : 1]
train_size, test_size


# In[18]:


train_data


# In[ ]:


# The above data is related to time series Dataset's


# In[25]:


def create_dataset(dataset, time_step = 1):
    X_data, Y_data = [], []
    for i in range(len(dataset) - time_step - 1):
	    a = dataset[i : (i + time_step), 0] 
	    X_data.append(a)
	    Y_data.append(dataset[i + time_step, 0])
    return np.array(X_data), np.array(Y_data)


# In[26]:


time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)


# In[27]:


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[28]:


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)


# In[39]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
from keras.models import Sequential 
from keras.layers import Dense, LSTM 


# In[38]:


# Now tarting LSTM Model
lst = Sequential()
lst.add(LSTM(50,return_sequences = True, input_shape = (100, 1)))
lst.add(LSTM(50, return_sequences = True))
lst.add(LSTM(50))
lst.add(Dense(1))
lst.compile(loss = 'mean_squared_error', optimizer='adam')


# In[41]:


lst.summary()


# In[42]:


lst.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100, batch_size = 64, verbose = 1)


# In[47]:


train_predict = lst.predict(X_train)
test_predict = lst.predict(X_test)


# In[ ]:


# The avove data shows the 1 and 2 prediction 


# In[48]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[49]:


look_back = 100
train_num_pyredict_plot = np.empty_like(data_close)
train_num_pyredict_plot[:, :] = np.nan
train_num_pyredict_plot[look_back : len(train_predict) + look_back, :] = train_predict
test_predict_plot = np.empty_like(data_close)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1 : len(data_close) - 1, :] = test_predict
plt.plot(scaler.inverse_transform(data_close))
plt.plot(train_num_pyredict_plot)
plt.plot(test_predict_plot)
plt.show()


# In[ ]:


Finally the concuslion done .

