import random
from typing import Callable

def make_overtaking_process(driver, constructor, course):
    return lambda opponent: random.random() > 0.5