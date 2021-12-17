from typing import Callable


def make_pit_stop_process(driver: str, constructor: str, course: str, year: int) -> Callable[[], bool]:
    return lambda: False


def make_pit_stop_duration_process(driver: str, constructor: str, course: str, year: int) -> Callable[[], float]:
    return lambda: 10
