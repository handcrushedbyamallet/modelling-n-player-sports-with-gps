
from typing import Callable


def make_pit_stop_process(driver, constructor, course):
    return lambda: False


def make_pit_stop_duration_process(driver, constructor, course):
    return lambda: 10
