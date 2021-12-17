import pandas as pd
import os
import glob
from typing import List, Optional, Any, Union


class F1Dataset:
    """API to Formula 1 data. This object facilitates the joining of multiple
    Formula 1 datasets as well as lazy-loading of the datasets to concerve
    memory and general utilities around the data.

        Args:
            dirpath (str): The directory with the F1 data files in
    """
    def __init__(self, dirpath: str):
        self.__loaded = []
        self.dirpath = dirpath
        self.datasets = [os.path.basename(fp).removesuffix('.csv') 
                         for fp in glob.glob(f'{dirpath}/*.csv')]
        self._data = {}

    def __getattr__(self, __name: str) -> pd.DataFrame:
        if __name not in self.datasets:
            raise AttributeError(f"F1Dataset has no attribute {__name}")
        else:  # Lazy load dataset
            if __name in self.__loaded:
                return self._data[__name]
            else:
                self._data[__name] = pd.read_csv(os.path.join(self.dirpath, __name + '.csv'))
                self.__loaded.append(__name)
                return self._data[__name]

    def __repr__(self) -> str:
        repr_str = 'F1Dataset with the following dataframes:\n'
        for dataset in self.datasets:
            repr_str += dataset + ':\n  ' 
            repr_str += '\n  '.join(self._data[dataset].columns)
            repr_str += '\n\n'
        return repr_str
        
    def driver_id_to_name(self, driver_id: int) -> str:
        """Return a driver name from the driverId number in the dataset

        Args:
            driver_id (int): The driver ID in the F1Dataset

        Returns:
            str: The driver's name
        """
        return self.__driver_mapping[driver_id]

    def constructor_id_to_name(self, constructor_id: int) -> str:
        """Return a constructor name from the constructorId number in the dataset

        Args:
            constructor_id (int): The constructor ID in the F1Dataset

        Returns:
            str: The constructor's name
        """
        return self.__constructor_mapping[constructor_id]

    def course_id_to_name(self, course_id: int) -> str:
        """Return a course name from the courseId number in the dataset

        Args:
            course_id (int): The course ID in the F1Dataset

        Returns:
            str: The course's name
        """
        return self.__course_mapping[course_id]

