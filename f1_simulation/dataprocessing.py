import pandas as pd
import os
from typing import List, Optional


class F1Dataset:
    """API to Formula 1 data. This object facilitates the joining of multiple
    Formula 1 datasets.
    """
    def __init__(self, dirpath: str, datasets: Optional[List[str]] = None):
        data = {}
        if datasets is None:
            self.datasets = [
                "circuits",
                "constructor_results",
                "constructor_standings",
                "constructors",
                "driver_standings",
                "drivers",
                "lap_times",
                "pit_stops",
                "qualifying",
                "races",
                "results",
                "seasons",
                "status"
            ]
        else:
            self.datasets = datasets

        for dataset in self.datasets:
            data[dataset] = pd.read_csv(os.path.join(dirpath, dataset + '.csv'))

        self._data = data

    def __getattr__(self, __name: str) -> pd.DataFrame:
        if __name not in self.datasets:
            raise AttributeError(f"F1Dataset has no attribute {__name}")
        else:
            return self._data[__name]

    def __repr__(self) -> str:
        repr_str = 'F1Dataset with the following dataframes:\n'
        for dataset in self.datasets:
            repr_str += dataset + ':\n  ' 
            repr_str += '\n  '.join(self._data[dataset].columns)
            repr_str += '\n\n'
        return repr_str
        