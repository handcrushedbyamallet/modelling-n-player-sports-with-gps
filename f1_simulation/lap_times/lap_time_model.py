import numpy as np


def make_lap_time_process(driver, constructor, course):
    return lambda: np.random.normal(loc=1200, scale=100)

