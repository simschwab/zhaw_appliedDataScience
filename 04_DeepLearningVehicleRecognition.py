# Databricks notebook source
# MAGIC %md <h1>Deep Learning Model to recognise vehicles</h1>

# COMMAND ----------

#installing tensorflow
#!pip install tensorflow

# COMMAND ----------

# Folder in which the pictures are stored
mountmap="100vehiclessmall"


# LOAD libraries
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import os
from PIL import Image

# COMMAND ----------

containername="vehicles100"

#LOAD data
n_train_data = 800
print("Number of traindata: " + str(n_train_data))
                   
n_test_data = 100
print("Number of testdata: " + str(n_test_data))
                  
#set categories
categoriesList=['ambulance', 'bicycle', 'bus', 'car', 'limousine', 'motorcycle', 'tank', 'taxi', 'truck', 'van']
categoriesSet={"ambulance":0 ,"bicycle":1 ,"bus":2, "car":3, "limousine":4, "motorcycle":5, "tank":6, "taxi":7, "truck":8, "van":9}


def load_via_dir(directory, n_max_datapoints, n_test):
    print("Loading data of path: " + directory)
    data_train_picture=[]
    data_train_label=[]  
    data_test_picture=[]
    data_test_label=[] 
    i=0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"): 
            #print(os.path.join(directory, filename))
            picture = np.asarray(Image.open(os.path.join(directory, filename)))
            name, numpng = filename.split("_",2)
            num, png = numpng.split(".",1) 
            n_max_train = (n_max_datapoints - n_test)/len(categoriesSet)
            if int(num) < n_max_train + 1000:        
                data_train_picture.append(picture)
                data_train_label.append(categoriesSet[name])
            else:
                data_test_picture.append(picture)
                data_test_label.append(categoriesSet[name])        
        
            if i % 100 == 0 and i != 0:
                print(str(i)+ " of " + str(n_train_data + n_test_data) + " pictures loaded.")
            i=i+1
        
            if(n_max_datapoints==len(data_train_picture) + len(data_test_picture)):
                print(str(len(data_train_picture) + len(data_test_picture))+" pictures loaded.")
                return data_train_picture, data_train_label, data_test_picture, data_test_label
            continue
        else:
              continue

# COMMAND ----------

data_train_picture, data_train_label, data_test_picture, data_test_label = load_via_dir("/dbfs/mnt/" + mountmap + "/", n_train_data+n_test_data, n_test_data)

# COMMAND ----------

# The data, shuffled and split between train and test sets:
x_train = np.asarray(data_train_picture)
y_train = np.asarray(data_train_label)
x_test = np.asarray(data_test_picture)
y_test = np.asarray(data_test_label)



print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_test)


# COMMAND ----------

#1b. Initial Model Training ParametersParameter Setup

batch_size = 32
num_classes = 10
epochs = 30

# COMMAND ----------

#1c. Preprocessing Steps
#1. Convert class vectors to binary class matrices
#2. Cast PIXEL Values to FLOAT
#3. Normalize Pixel RGB Values to (0:1)

#1
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
#2
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#3
x_train /= 255
x_test /= 255

# COMMAND ----------

print(y_test)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md ##### 1d. Define Convolution Network

# COMMAND ----------

def initializeModel():
  model = Sequential()

  model.add(Conv2D(32, (3, 3), padding='same',
                   input_shape=x_train.shape[1:]))
                   
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  return model

# COMMAND ----------

# COMMAND ----------

model = initializeModel()

# COMMAND ----------

# COMMAND ----------

## Print Model Summary
model.summary()

# COMMAND ----------

# COMMAND ----------

# MAGIC %md ##### 1e. Define Optimizers and Compile the model
# MAGIC 
# MAGIC **Optimization algorithms** are used to minimize (or maximize) an Objective function (also called Error function) E(x) which is mathematical function dependent on the Model’s internal learnable parameters that are used in computing the target values(Y) from the set of predictors(X) used by the machine learning model. In neural networks the Weights(W) and the Bias(b) values are the learnable parameters which are used in computing the output values and are learned and updated in the direction of best solution i.e minimizing the Loss by the network’s training process.
# MAGIC 
# MAGIC Below are som of the most used optimizers available in Keras:
# MAGIC 1. Stochastic Gradient Descent (SGD)
# MAGIC 2. Root Mean Square Propagation (RMSProp)
# MAGIC 3. Adaptive Gradient (ADAGrad)
# MAGIC 4. Adaptive Moment Estimation (ADAM)
# MAGIC   
# MAGIC **Hyper Parameters**
# MAGIC   Two hyper parameters used by most optimizers are:
# MAGIC   1. Learning Rate - Learning rate (lr) controls the magnitude of adjustment of the weights of the neural network with respect the loss gradient.
# MAGIC   2. Decay - Weight decay is a regularization term that causes weights to exponentially decay to zero and hence penalizes big weights.

# COMMAND ----------

# initiate RMSprop optimizer
from tensorflow.keras import optimizers
import tensorflow as tf


opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# COMMAND ----------

# COMMAND ----------

# MAGIC %md ##### 1f.  Model Training : without Data Augmentation (Takes 1 Min)

# COMMAND ----------

from keras.callbacks import History 
history_noAug = History()
print('Not using data augmentation.')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
         callbacks=[history_noAug],
         verbose=1)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md ##### 1g. Evaluate model

# COMMAND ----------

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
print('Test loss:', score[0])

# COMMAND ----------

# COMMAND ----------

def plotmetrics(history):
  width = 10
  height = 5
  ## Clear plot if repeated call
  plt.clf()
  plt.figure(figsize=(width, height))
  # Plot training & validation accuracy values
  plt.title('Model Metrics : Non Augmented Data')
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  # Plot training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  pltoutput = plt.show()
  return pltoutput


# COMMAND ----------

# COMMAND ----------

# Visualize model
pltoutput = plotmetrics(history_noAug)
display(pltoutput)

# COMMAND ----------

# MAGIC %md ##### 1h. Show predictions in plot

# COMMAND ----------

#model.save(r'C:\Users\mario\OneDrive\Desktop\model_smallpictures_100.h5')

# COMMAND ----------

#1i. Save Trained Model (without Data Augmentation) to DBFS as HDF5 file

## Create Output Model Directory
dbutils.fs.mkdirs('/vehicles/models/')

# COMMAND ----------

#save model
model.save('/tmp/vehicles_100.h5')

dbutils.fs.cp("file:/tmp/vehicles_100.h5", "dbfs:/tmp/vehicles_100.h5")
display(dbutils.fs.ls("/tmp/vehicles_100.h5"))

# COMMAND ----------

# COMMAND ----------

from keras.models import load_model
modelpath = '/dbfs/tmp/vehicles_100.h5'
#model.save(modelpath)

model = load_model(modelpath)

# COMMAND ----------

# COMMAND ----------

pltoutput = plotmetrics(history_noAug)
display(pltoutput)


# COMMAND ----------

#2h. Show predictions in plot

categoriesList=['ambulance', 'bicycle', 'bus', 'car', 'limousine', 'motorcycle', 'tank', 'taxi', 'truck', 'van']
#categoriesList = np.array(categoriesList)

import matplotlib.pyplot as plt
import random
def plotImages(x_test, images_arr, labels_arr, n_images=8):
    fig, axes = plt.subplots(n_images, n_images, figsize=(9,9))
    axes = axes.flatten()
    
    for i in range(100):
        rand = random.randint(0, x_test.shape[0] -1)
        img = images_arr[rand]
        ax = axes[i]
    
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
        
        #predictions = model.predict_classes([[x_test[rand]]])
        #predictions = model.predict([[x_test[rand]]])
        predictions = np.argmax(model.predict(x_test), axis=-1)
 
        label=categoriesList[predictions[0]]   
        
        print(predictions[0])
        print(labels_arr[rand])
        
        if labels_arr[rand][predictions[0]] == 0:
            ax.set_title(label, fontsize=18 - n_images, color="red") 
        else:
            ax.set_title(label, fontsize=18 - n_images) 
        
    plot = plt.tight_layout()
    return plot

  
display(plotImages(x_test, x_test, y_test, n_images=10))