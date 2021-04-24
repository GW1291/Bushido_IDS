import logging
import pandas as pd
import pytest

class ProcessedData():
    def __init__(self):
        ##vars
        self._labels = None


class Data():
    default_csv_path = "data/corrected.csv"
    def __init__(self,csv_path:str=None):
        ##vars
        self._csv_path = csv_path
        self._dataframe = None
        self._logger = None
        self._tensor = None

        ##logic
        self._set_default_logger()
        self._logger.debug(f'{self._set_default_logger.__annotations__}')

        if not self._csv_path:
            self._csv_path = self.default_csv_path
            self._logger.debug(f'csv_path={self._csv_path}')

        self._set_default_dataframe(self._csv_path)
        self._logger.debug(f'{self._set_default_dataframe.__annotations__}')

    
    def _set_default_logger(self) -> 'assign':
        self._logger = logging.getLogger()

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