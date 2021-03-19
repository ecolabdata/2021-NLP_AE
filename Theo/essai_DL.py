#%%
import pandas as pd
import numpy as np
import sklearn
import pickle
import tensorflow as tf 
import keras
import torch
import gc
import psutil
#%%
psutil.cpu_percent(interval=0.5)
int(psutil.virtual_memory().total - psutil.virtual_memory().available)/ 1024 / 1024 / 1024
#%%
try:
    
    del X_train
    del X_test
    del Y_train
    del Y_test
    del model
except:
    print("Il n'y a pas les éléments à supprimer.")

gc.collect()
#%%
p=100
N1=800000
# N2=50000

features_1=np.random.rand(N1,50,p)
#%%
Y=np.random.rand(N1,50,1)

# features=
# df_features=pd.concat([pd.DataFrame(features_1),pd.DataFrame(features_2)])
frac=int(N1*(4/5))
Y_train=Y[:frac]
X_train=features_1[:frac]
Y_test=Y[frac:]
X_test=features_1[frac:]
del features_1,Y
print(X_train.shape,Y_train.shape)
#%%
from keras.layers import Input,Conv1DTranspose, Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, Dense, Dropout, Activation, Lambda, Reshape, Flatten
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.metrics import RootMeanSquaredError
#%%
model = Sequential()
callback = EarlyStopping(monitor='val_loss', patience=10)
# v=
# u=
# Recurrent layer
u1=500
model.add(Dense(u1,kernel_regularizer=l2(1e-3),input_shape=(X_train.shape[1],X_train.shape[2])))

k1=15
# model.add(Conv1D(1,k1,activation='sigmoid',kernel_regularizer=l2(1e-3)))

model.add(Dropout(0.5))
model.add(Dense(250, activation='sigmoid'))
model.add(Conv1D(150,k1,activation='sigmoid'))
# ,kernel_regularizer=l2(1e-3)
# Output layer
#model.add(LSTM(5000, return_sequences=True))
#model.add(Dense(5000,activation='sigmoid'))
model.add(Dropout(0.5))
# model.add(K.expand_dims())
k2=10
model.add(Conv1D(80,k2,activation='sigmoid'))
#model.add(LSTM(1000, return_sequences=True))
model.add(Dense(30,activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(LSTM(100, return_sequences=True))
#model.add(Dense(1,activation='sigmoid'))

#model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(200,activation='sigmoid')))
#model.add(Dropout(0.5))
model.add(Conv1DTranspose(1,k1+k2-1))
# model.add(Conv1D(1,u1))
#model.add(Dense(1))
#model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mae',metrics=['mae'])
model.summary()
# %%
import time
e=300
start_time = time.time()
history = model.fit(X_train, Y_train,epochs=e,validation_data=(X_test,Y_test),verbose=0,
                    callbacks=[callback])
end_time = time.time()
print('Le calcul a pris :',(end_time - start_time)/60,'minutes')
# %%
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, e+1)
l=len(loss_values)
plt.plot(epochs[:l], loss_values, 'bo', label='Training loss')
plt.plot(epochs[:l], val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
