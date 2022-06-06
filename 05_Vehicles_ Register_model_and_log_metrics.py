# Databricks notebook source
# MAGIC %md ##0. Set parameters

# COMMAND ----------

workspace="zhawadsMachineLearning"
resource_grp="zhaw_AppliedDataScience"
subscription_id="9d766322-e0aa-4e0d-9225-aa934bcc1b4a"



path= '/dbfs/tmp/'
par_model_name = 'vehicles_100.h5'
#par_modelall_name = 'cifar_allpictures.h5' 

par_experiment_name = 'vehiclesrecognition'

# In case cell gets status "cancelled" after execution, uninstall libraries, restart cluster and reinstall libraries

# COMMAND ----------

# MAGIC %md ##1.  Log metrics of models

# COMMAND ----------

# MAGIC %md ##### 1a. Authenticate to Azure ML workspace

# COMMAND ----------

import sys
import requests
import time
import base64
import datetime
import azureml.core
import shutil
import os, json
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.model import Model
import azureml.core
from azureml.core.authentication import ServicePrincipalAuthentication

ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp)

ws.get_details()

# COMMAND ----------

# MAGIC %md ##### 1b. Load models from disk where it was stored

# COMMAND ----------

import keras
from keras.models import load_model

path= '/dbfs/tmp/'
modelpath = path + par_model_name
#modelallpath = path + par_modelall_name

model = load_model(modelpath)
#modelall = load_model(modelallpath)

# COMMAND ----------

model.summary()

# COMMAND ----------

# MAGIC %md ##### 1c. Get testdata to regenerate metrics

# COMMAND ----------

from keras.utils import np_utils
#from keras.datasets import cifar10
num_classes = 10
# The data, shuffled and split between train and test sets:



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
#2
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#3
x_train /= 255
x_test /= 255

# COMMAND ----------

# MAGIC %md ##### 1d. Create new experiment in Azure ML service workspace

# COMMAND ----------

# start a training run by defining an experiment
myexperiment = Experiment(ws, par_experiment_name)
run = myexperiment.start_logging()
run = myexperiment


#run.complete()
run_id = run.id
print ("run id:", run_id)

# COMMAND ----------

# MAGIC %md ##2. Register the model

# COMMAND ----------

registermodelall = Model.register(
    model_path=modelpath,  # this points to a local file
    model_name=par_model_name,  # this is the name the model is registered as
    tags={"area": "spark", "type": "deeplearning", "run_id": run_id},
    description="Keras deeplearning, vehicles recognition",
    workspace=ws,
)
print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(registermodelall.name, registermodelall.description, registermodelall.version))