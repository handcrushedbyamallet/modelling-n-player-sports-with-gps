from typing import Callable


def make_pit_stop_process(driver, constructor, course) -> Callable[[], bool]:
    return lambda: False


def make_pit_stop_duration_process(driver, constructor, course) -> Callable[[], float]:
    return lambda: 10
