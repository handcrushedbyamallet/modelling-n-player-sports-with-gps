import numpy as np
from typing import Callable


def make_lap_time_process(driver: str, constructor: str, course: str) -> Callable[[], float]:
    return lambda: np.random.normal(loc=1200, scale=100)

