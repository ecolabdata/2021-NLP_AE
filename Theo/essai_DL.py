################################################################################################################################################################################
########### Essais et tests de Deep Learning ########################################################################################################################################################################
################################################################################################################################################################################
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
try:
    del X_train
    del X_test
    del Y_train
    del Y_test
    del model
except:
    print("Il n'y a pas les éléments à supprimer.")

gc.collect()
psutil.cpu_percent(interval=0.5)
int(psutil.virtual_memory().total - psutil.virtual_memory().available)/ 1024 / 1024 / 1024

#%%
p=500
n=3000
U=1000
Input_shape=(U,n,p)
# features=tf.random.normal(Input_shape)
features=np.random.rand(Input_shape)
#%%
Input_shape_y=(U,n,1)
Y=np.random.rand(Input_shape_y)
#%%
frac=int(U*(4/5))
Y_train=Y[:frac]
X_train=features[:frac]
Y_test=Y[frac:]
X_test=features[frac:]
del features,Y
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

u1=1000
model.add(Dense(u1,kernel_regularizer=l2(1e-3),input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(300, activation='sigmoid'))

k1=1000
model.add(Conv1D(150,k1,activation='sigmoid'))
model.add(Dropout(0.5))

k2=1000
model.add(Conv1D(80,k2,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Conv1DTranspose(1,k1+k2-1,activation='sigmoid'))

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
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, e+1)
l=len(loss_values)
print('Le calcul a pris :',(end_time - start_time)/60,'minutes')
print('Il y a eu :',l,'époques.')
# %%

import matplotlib.pyplot as plt
plt.plot(epochs[:l], loss_values, 'bo', label='Training loss')
plt.plot(epochs[:l], val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
################################################################################################################################################################################
########### Essais TensorFlow Parallelization ########################################################################################################################################################################
################################################################################################################################################################################
try:
    del X_train
    del X_test
    del Y_train
    del Y_test
    del model
except:
    print("Il n'y a pas les éléments à supprimer.")

print(gc.collect())
psutil.cpu_percent(interval=0.5)
print(int(psutil.virtual_memory().total - psutil.virtual_memory().available)/ 1024 / 1024 / 1024)


p=500
n=3000
U=1000
Input_shape=(U,n,p)
features=tf.random.normal(Input_shape)
Input_shape_y=(U,n,1)
Y=tf.random.uniform(Input_shape_y,minval=0,maxval=1)
frac=int(U*(4/5))
Y_train=Y[:frac]
X_train=features[:frac]
Y_test=Y[frac:]
X_test=features[frac:]
del features,Y
print(X_train.shape,Y_train.shape)
#%%
strategy=tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

units=np.array([1000,300,30])
filters=np.array([150,80])
kernels=[1000,1000]
dropout=np.repeat(0.5,3)
#%%
with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(units[0], activation='relu', input_shape=X_train.shape[1:]),
      tf.keras.layers.Dropout(dropout[0]),
      tf.keras.layers.Dense(units[1],activation="relu"),
      tf.keras.layers.Conv1D(filters[0],kernels[0],activation="sigmoid"),
      tf.keras.layers.Dropout(dropout[1]),
      tf.keras.layers.Conv1D(filters[1],kernels[1],activation="sigmoid"),
      tf.keras.layers.Dense(units[2], activation='relu'),
      tf.keras.layers.Conv1DTranspose(1,int(np.sum(kernels)-1),activation="sigmoid")])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# %%
# Define the checkpoint directory to store the checkpoints

checkpoint_dir = 'C:/Users/theo.roudil-valentin/Documents/Donnees/essai_DL_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# %%
# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5
#%%
# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
#%%
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
#%%
model.summary()
# %%
import time
e=12
start_time = time.time()
history = model.fit(X_train, Y_train,epochs=e,#validation_data=(X_test,Y_test),
                    callbacks=callbacks)
end_time = time.time()
print('Le calcul a pris :',(end_time - start_time)/60,'minutes')

# %%
