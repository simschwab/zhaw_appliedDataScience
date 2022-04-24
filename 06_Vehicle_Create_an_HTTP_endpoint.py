# Databricks notebook source
# MAGIC %md Azure ML & Azure Databricks notebooks by René Bremer (original taken from Parashar Shah)
# MAGIC 
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md ##### In this notebook the following steps will be excuted:
# MAGIC 
# MAGIC 1. Create endpoint of best model (trained with 60000 pictures)
# MAGIC 
# MAGIC Make sure you added libraries to azureml-sdk[databricks], Keras and TensorFlow to your cluster.

# COMMAND ----------

# MAGIC %md #0. Set parameters

# COMMAND ----------

workspace="zhawadsMachineLearning"
resource_grp="zhaw_AppliedDataScience"
subscription_id="9d766322-e0aa-4e0d-9225-aa934bcc1b4a"


par_model_name = 'vehicles_100.h5' 
par_service_name = 'vehicles'

# In case cell gets status "cancelled" after execution, uninstall libraries, restart cluster and reinstall libraries

# COMMAND ----------

# MAGIC %md #1. Create endpoint of best model (trained with 60000 pictures)

# COMMAND ----------

# MAGIC %md ##### 1a. Authenticate to Azure ML workspace (interactive, using AAD and browser)

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

# MAGIC %md ##### 1b. Retrieve best model from Azure ML Service

# COMMAND ----------

import keras
from keras.models import load_model

path= '/dbfs/tmp/'
modelpath = path + par_model_name
#modelallpath = path + par_modelall_name

model = load_model(modelpath)
#modelall = load_model(modelallpath)


#model=Model(ws,par_model_name)
model=model
model_list = Model.list(workspace=ws)
print("Model picked: {} ".format(model.name))

# COMMAND ----------

# MAGIC %md ##### 1c. Create score file (script that will be used in endpoint to consume png) and conda env

# COMMAND ----------

#%%writefile score_deeplearning.py
score_deeplearning = """

import json

from azureml.core.model import Model
from keras.models import load_model
from io import BytesIO
import numpy as np
from PIL import Image
from base64 import b64decode

def init():
    global trainedModel
    # retreive the path to the model file using the model name
    # This needs to be the name of your model you registered in EstimatorTrigger.py
    print("Load model")
    model_name = "{model_name}"  # interpolated
    model_path = Model.get_model_path(model_name)
    trainedModel = load_model(model_path)
    print("model loaded")

def run(raw_data):
    print("base64 picture received")
    imagebase64=json.loads(raw_data)['imagebase64']
    img = Image.open(BytesIO(b64decode(imagebase64)))
    new_img = white_bg_square(img)
    resized_img=new_img.resize((32, 32), Image.ANTIALIAS)
    x_data = np.asarray(resized_img)
    x_data = x_data.astype('float32')
    x_data /= 255    
    print("make prediction")
    input_data = []
    input_data.append(x_data)
    predictions = trainedModel.predict_classes([[input_data[0]]])

    categoriesList = ['ambulance', 'bicycle', 'bus', 'car', 'limousine', 'motorcycle', 'tank', 'taxi', 'truck', 'van']
    print("create label prediction")
    label=categoriesList[predictions[0]]
    print("label: " +  label)
    return json.dumps({{"result":label}})

def white_bg_square(img):
    "return a white-background-color image having the img in exact center"
    size = int(img.size[0]), int(img.size[1]) # (int(max(img.size)),)*2
    layer = Image.new('RGB', size, (255,255,255))
    imgsizeint = int(img.size[0]), int(img.size[1])
    layer.paste(img, tuple(map(lambda x:int((x[0]-x[1])/2), zip(size, imgsizeint))))
    return layer

""".format(model_name=par_model_name)

exec(score_deeplearning)

with open("score_deeplearning.py", "w") as file:
    file.write(score_deeplearning)
    


# COMMAND ----------

score_deeplearning.save('/dbfs/tmp/score_deeplearning.py')


# COMMAND ----------

score_deeplearning.save("file:\\\C:\\Users\\mario\\OneDrive\\Dokumente\\Mario\\score_deeplearning.py")

# COMMAND ----------

from azureml.core import Environment
service_env = Environment(name='service-env')
python_packages = ['scikit-learn', 'keras','numpy','Pillow'] # whatever packages your entry script uses

for package in python_packages:
    service_env.python.conda_dependencies.add_pip_package(package)

from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)
#from azureml.core.conda_dependencies import CondaDependencies 

#myacienv = CondaDependencies.create(conda_packages=['scikit-learn', 'keras','numpy','Pillow'])

#with open("deeplearningenv.yml","w") as f:
  #  f.write(myacienv.serialize_to_string())

# COMMAND ----------

# MAGIC %md ##### 1d. Deploy model and create endpoint

# COMMAND ----------

try:
    oldservice = Webservice(workspace=ws, name=par_service_name)
    print("delete " + par_service_name + " before creating new one")
    oldservice.delete()
except:
    print(par_service_name + " does not exist, create new one")

# COMMAND ----------

from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice, Webservice

image_config = ContainerImage.image_configuration(execution_script="score_deeplearning.py",
                                                  runtime="python",
                                                  conda_file="deeplearningenv.yml")

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 2,
    memory_gb = 4,
    tags = {'name':'Databricks ALM ACI'},
    description = 'AML Deployment Production')


# COMMAND ----------

service = Webservice.deploy_from_model(
  workspace=ws,
  name=par_service_name,
  deployment_config = aci_config,
  models = [model],
  image_config = image_config
    )

service.wait_for_deployment(show_output=True)

# COMMAND ----------

