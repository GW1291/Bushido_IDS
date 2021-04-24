import logging

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

import pytest

from pre_processing import Data

def convert_dataframe_dtypes(dataframe):
        logger = logging.getLogger()

        #all objects to dummy
        converted_dataframe_dummy = pd.get_dummies(dataframe,drop_first=True)
        logger.debug(f'{converted_dataframe_dummy.dtypes}')

        #all cast to float16
        converted_dataframe_dummy_float16 = converted_dataframe_dummy.astype('float16')
        logger.debug(f'{converted_dataframe_dummy_float16.dtypes}')

        #x_and_y conversion
        #y
        labels = pd.DataFrame()
        for column in converted_dataframe_dummy_float16.columns:
            if 'label' in column:
                label = converted_dataframe_dummy_float16.pop(column)
                logger.debug(f'popping label {column} {label.dtype}')
                labels[column] = label
        logger.info(f'Label dataframe\n {labels.dtypes}')

def test_convert_dataframe_dtypes():
    tData = Data()
    dataframe = tData._dataframe
    convert_dataframe_dtypes(dataframe)
