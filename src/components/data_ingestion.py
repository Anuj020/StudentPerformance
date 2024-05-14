import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation , DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
'''
This file is used for storing train path or test path or raw data, 
So those type of input basically created in one class.
Any input requires basically write down here.
'''
@dataclass 
# for class variable we use init, but when we use dataclass decorator, we can directly define class variable.
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # storing all the tree values od above class
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # here need to change if want to read data from different sources
            df = pd.read_csv('/Users/anuj/Desktop/MlProject/Notebook/stud.csv')   
            logging.info('Exported the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # check if directory is alredy created if not, will create it.
        
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)  
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))