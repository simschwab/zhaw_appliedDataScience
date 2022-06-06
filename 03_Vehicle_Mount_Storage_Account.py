# Databricks notebook source
# MAGIC %md <h1>Mount Storage Account to access vehicle images</h1>

# COMMAND ----------

# MAGIC %md <h5>Set parameters</h5>

# COMMAND ----------

import config

storageaccount = config.azure_storageaccount

account_key= config.azure_account_key

containername="vehicles"
mountname = "vehiclessmall"

# COMMAND ----------

# MAGIC %md <h5>1. Mount blob storage</h5>

# COMMAND ----------

if any(mount.mountPoint == "/mnt/" + mountname for mount in dbutils.fs.mounts()):
  print ("directory " + "/mnt/" + mountname + " is already mounted")
else:
  print ("In case you have a cluster with 0 workers, you need to cancell statement manually after 30 seconds. This is because a spark job is started, which cannot be executed since there are 0 workers. However, the storage is mounted, which can be verified by rerunning cell")
  dbutils.fs.mount(
  source = "wasbs://" + containername + "@" + storageaccount + ".blob.core.windows.net",
  mount_point = "/mnt/" + mountname,
  extra_configs = {"fs.azure.account.key." + storageaccount +".blob.core.windows.net":account_key})

# COMMAND ----------

# MAGIC %md <h5>2. Unzip pictures in storage account</h5>

# COMMAND ----------

#2. unzip data

import zipfile
import os


datafile = "vehicles.zip"

datafile_dbfs = os.path.join("/dbfs/mnt/" + mountname, datafile)

zip_ref = zipfile.ZipFile(datafile_dbfs, 'r')
zip_ref.extractall("/dbfs/mnt/" + mountname)
zip_ref.close()


# COMMAND ----------

# MAGIC %md  <h5>3. List and show pictures</h5>

# COMMAND ----------

#3. Show pictures
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import os

categoriesList=['ambulance', 'bicycle', 'bus', 'car', 'limousine', 'motorcycle', 'tank', 'taxi', 'truck', 'van']

def plotImagesMount(n_images=8):
    fig, axes = plt.subplots(n_images, n_images, figsize=(9,9))
    axes = axes.flatten()
    
    for i in range(n_images * n_images):
        rand1 = random.randint(0, 5)
        rand2 = random.randint(1020, 1030)
        filename=str(categoriesList[rand1]) + "_" + str(rand2)
        filenamejpg=filename + ".jpg"
        img = Image.open(os.path.join("/dbfs/mnt/" + mountname + "/", filenamejpg))
        ax = axes[i]
    
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
        
        ax.set_title(filename, fontsize=18 - n_images)
        
    plot = plt.tight_layout()
    return plot
  
display(plotImagesMount(n_images=5))