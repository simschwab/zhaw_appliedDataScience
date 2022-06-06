# zhaw_appliedDataScience
  
Case: Recognize vehicle types for automatic toll classification
## Structure of the repository
### Collect Data
01_GetDataFromAPI  
01_ScrapeDataFromWebsite
### Prepare Data
02_1_Rename  
02_2_FindOptimalImageSizeForCNN
### Storage in Azure
03_Vehicle_Mount_Storage_Account
### Modeling
04_DeepLearningVehicleRecognition
### Analyze
05_Vehicles_Register_Model_and_log_metrics  
06_Vehicle_create_an_HTTP_endpoint
## Architecture Overview


## Tech-Stack

Azure Storage: Storage for pictures  
https://azure.microsoft.com/de-de/services/storage/blobs/

Azure Databricks: train model    
https://azure.microsoft.com/de-de/services/databricks/

Azure Machine Learning: Deployment of the service   
https://azure.microsoft.com/de-de/services/machine-learning/#product-overview


## Example Case:
Recognize classes of pictures (plane, frog, ships) using the cifar-10 dataset  
https://towardsdatascience.com/how-to-create-your-own-deep-learning-project-in-azure-509660d8297
