import random
from typing import Callable

def make_overtaking_process(driver: str, constructor: str, course: str, year: int):
    return lambda opponent: random.random() > 0.5