# Databricks notebook source
# MAGIC %md <h1>Mount Storage Account to access vehicle images</h1>

# COMMAND ----------

# 1. Mount blob storage
# 2. Unzip pictures in storage account
# 3. List and show pictures


# COMMAND ----------

# MAGIC %md Set parameters

# COMMAND ----------

storageaccount="zhawadsstorage"

account_key="BEPIWVQrb1Iw4rwPDe9oyzuwwF4MtTFL/uvmrMqPPD7T7jgi5s0dr8W5axdT3LlP4mCFwLOBLKo3+AStKTvmSw=="

containername="vehicles100"
#mountname=containername
mountname = "100vehiclessmall"

# COMMAND ----------

#1. Mount blob storage

if any(mount.mountPoint == "/mnt/" + mountname for mount in dbutils.fs.mounts()):
  print ("directory " + "/mnt/" + mountname + " is already mounted")
else:
  print ("In case you have a cluster with 0 workers, you need to cancell statement manually after 30 seconds. This is because a spark job is started, which cannot be executed since there are 0 workers. However, the storage is mounted, which can be verified by rerunning cell")
  dbutils.fs.mount(
  source = "wasbs://" + containername + "@" + storageaccount + ".blob.core.windows.net",
  mount_point = "/mnt/" + mountname,
  extra_configs = {"fs.azure.account.key." + storageaccount +".blob.core.windows.net":account_key})

# COMMAND ----------

#2. unzip data

import zipfile
import os

#datafile = "vehicledata.zip"
#with 100 pictures per categorie
datafile = "trainingdata100.zip"

datafile_dbfs = os.path.join("/dbfs/mnt/" + mountname, datafile)

zip_ref = zipfile.ZipFile(datafile_dbfs, 'r')
zip_ref.extractall("/dbfs/mnt/" + mountname)
zip_ref.close()


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
        rand1 = random.randint(0, 6)
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

# COMMAND ----------

