import logging

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

import tensorflow as tf

import keras.backend as K

import pytest

class Logger():
    def __init__(self):
        self._set_default_logger()
        self._logger.debug(f'{self._set_default_logger.__annotations__}')

    def _set_default_logger(self) -> 'assign':
        self._logger = logging.getLogger()
    
    @property
    def logger(self):
        return self._logger

class ProcessedData():
    def __init__(self,unformatted_dataframe):
        ##logging composition
        self._logger = Logger().logger
        ##vars
        self._labels_len = None
        self._features_len = None

        self._unformatted_dataframe = unformatted_dataframe
        self._formatted_dataframe = None
        
        self._feature_dataframe = None
        self._label_dataframe = None

        self._X_train_dataframe = None
        self._X_test_dataframe = None

        self._y_train_dataframe = None
        self._y_test_dataframe = None

        self._training_dataset = None
        self._testing_dataset = None

        ##logic
        self._formatted_dataframe = (
            self._convert_dataframe_strs_to_dummy(self._unformatted_dataframe))
        
        self._label_dataframe,self._feature_dataframe = self._create_label_feature_dataframes(self._formatted_dataframe)

        self._features_len = len(self._feature_dataframe.columns)
        self._labels_len = len(self._label_dataframe.columns)

        self._X_train_dataframe, self._X_test_dataframe, self._y_train_dataframe, self._y_test_dataframe = self._split_dataframe_20_80(self._feature_dataframe,self._label_dataframe)

        self._training_dataset = self._create_tensorflow_dataset(self._X_train_dataframe,self._y_train_dataframe)

        self._testing_dataset = self._create_tensorflow_dataset(self._X_test_dataframe,self._y_test_dataframe)

        #self._convert_dataframe_to_tensor(self._label_dataframe)
        #self._convert_dataframe_to_tensor(self._feature_dataframe)

    def _convert_dataframe_strs_to_dummy(self,dataframe):
        converted_dataframe_dummy = pd.get_dummies(dataframe,drop_first=True)
        self._logger.debug(f'{converted_dataframe_dummy.dtypes}')
        return converted_dataframe_dummy

    #TODO: deprecate
    def _convert_dataframe_dtype_float16(self,dataframe):
        converted_dataframe_dummy_float16 = dataframe.astype('float16')
        self._logger.debug(f'{converted_dataframe_dummy_float16.dtypes}')
        #return converted_dataframe_dummy_float16
        return dataframe

    def _create_label_feature_dataframes(self,dataframe)-> tuple:
        label_dataframe = pd.DataFrame()
        for column in dataframe.columns:
            if 'label' in column:
                label = dataframe.pop(column)
                self._logger.debug(f'popping label {column} {label.dtype}')
                label_dataframe[column] = label
        self._logger.info(f'Label dataframe\n {label_dataframe.dtypes}')
        self._logger.info(f'Feature dataframe\n {dataframe.dtypes}')
        feature_dataframe = dataframe
        return label_dataframe,feature_dataframe

    def _convert_dataframe_to_tensor(self,dataframe):
        #tensor = tf.data.Dataset.from_tensor_slices((dataframe.values))
        tensor = K.constant(dataframe.values)
        self._logger.debug(f'Label dataframe\n {K.eval(tensor)}')
        return tensor
    
    def _split_dataframe_20_80(self,feature_dataframe,label_dataframe):
        X_train, X_test, y_train, y_test = train_test_split(feature_dataframe, label_dataframe, test_size=0.20, random_state=42)
        return  X_train, X_test, y_train, y_test 
    
    def _create_tensorflow_dataset(self,X_dataframe,y_dataframe):
        dataset = tf.data.Dataset.from_tensor_slices((X_dataframe.values,y_dataframe.values))
        self._logger.debug(f'created tensorflow dataset\n {dataset}')
        return dataset 

class Data():
    default_csv_path = "data/corrected.csv"
    def __init__(self,csv_path:str=None):
        ##logging composition
        self._logger = Logger().logger

        ##vars
        self._csv_path = csv_path
        self._dataframe = None

        ##logic
        if not self._csv_path:
            self._csv_path = self.default_csv_path
            self._logger.debug(f'csv_path={self._csv_path}')

        self._set_default_dataframe(self._csv_path)
        self._logger.debug(f'{self._set_default_dataframe.__annotations__}')

        ##composition
        self.ProcessedData = ProcessedData(self._dataframe)

    def _set_default_dataframe(self,csv_path:str) -> 'assign':
        self._dataframe = self._create_dataframe(csv_path)

    def _create_dataframe(self,csv_path:str) -> 'dataframe':
        return pd.read_csv(csv_path)

    @property
    def dtypes(self):
        dtypes = self._dataframe.dtypes
        self._logger.info(f'imported csv dtypes:\n{dtypes}')
        return dtypes

@pytest.fixture(scope='module')
def TestDataFixture():
    class persist():
        pass
    persist = persist()
    return persist
class TestData():
    def test_init(self,TestDataFixture):
        TestDataFixture.tData = Data()
    def test_dtypes(self,TestDataFixture):
        TestDataFixture.tData.dtypes