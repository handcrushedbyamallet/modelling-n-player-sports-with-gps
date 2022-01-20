from __future__ import annotations

import lap_times
import pit_stopping
import overtaking

import datetime
from timeit import default_timer as timer
import pandas as pd
import numpy as np

def time_call(func, *args, **kwargs):
    before = timer()
    result = func(*args, **kwargs)
    after = timer()
    print(f"Function {func.__name__} took {after - before} seconds")
    return result

def get_seconds_from_timedelta(time):
    return time / np.timedelta64(1, 's')

class F1Racer:
    """The F1Racer class is the functioning heart of this simulation. It
    represents a single driver, constructor, car combination and stores the
    models for each of the different subproblems for the system. Upon
    initialisation it fits models to each subproblem. These systems are used to
    govern how long the racer takes to finish each lap.
    Args:
        driver_id (str): the name of the driver
        constructor_id (str): the name of the constructor
        course_id (str): The name of the course
        year (int): The year the race is occuring
        start_time (float): The time penalty incurred from starting in a later position
    """
    def __init__(
        self, 
        race_id: str,
        driver_id: str, 
        constructor_id: str, 
        course_id: str, 
        year: int, 
        starting_time: float,
        total_laps: int,
        top_quali: datetime.timedelta,
        overtaking_data: pd.DataFrame
    ):
        self.race_id = race_id
        self.driver = driver_id
        self.constructor = constructor_id
        self.course = course_id
        self.current_time = starting_time
        self.year = year
        self.laps_since_pit_stop = 0
        self.overtaking_data = overtaking_data
        time_call(self.initialise_lap_time_params, driver_id, year, total_laps, top_quali)
        time_call(self.initialise_overtake_params)
        time_call(self.initialise_pit_stop_params)

        self.overtaking_mode = None
        self.pit_stopping = None
        self.pit_stop_duration = None
        self.sampled_lap_time = None

    def __repr__(self):
        return f"""
        {self.driver=}
        {self.constructor=}
        {self.course=}
        {self.current_time=}
        {self.year=}
        {self.laps_since_pit_stop=}"""

    def initialise_lap_time_params(self, driver_id: int, year: int, total_laps: int, top_quali: datetime.timedelta, normalise_pit_laps: bool = True):
        """Fits the model that will govern the lap times of the racer
        """
        self.lap_time_process = lap_times.make_lap_time_process(driver_id=driver_id, year=year, total_laps=total_laps, top_quali=top_quali)

    def initialise_overtake_params(self):
        """Fits the model that will govern the racer's ability to overtake
        """
        self.overtake_process = overtaking.make_overtaking_process(driver=self.driver, constructor=self.constructor, courseId=self.course, year=self.year, df=self.overtaking_data)

    def initialise_pit_stop_params(self):
        """Fits the model that will govern the racer's need to pit stop as well
        as the duration of pit stops
        """
        self.pit_stop_process = pit_stopping.make_pit_stop_process(driver_id=self.driver, constructor_id=self.constructor, course_id=self.course, year=self.year)
        self.pit_stop_duration_process = pit_stopping.make_pit_stop_duration_process(driver_id=self.driver, constructor_id=self.constructor, course_id=self.course, year=self.year)

    def sample_lap_time(self, lap_number) -> float:
        """Sample a lap time from the racer's lap time model and the course

        Returns:
            float: The time taken to complete the lap in milliseconds.
        """
        return self.lap_time_process(lap_number, self.laps_since_pit_stop)

    def sample_overtake(self, opponent: F1Racer) -> bool:
        """Sample whether this racer can overtake the `opponent` racer

        Args:
            opponent (F1Racer): The racer who may be overtaken

        Returns:
            bool: Whether this racer successfully overtook the `opponent` racer
        """
        return self.overtake_process(opponent)

    def sample_pit_stop(self, car_before: float, car_after: float, lap_time: float) -> bool:
        """Samples whether the racer will go in for a pit stop

        Returns:
            bool: Whether the car goes in for a pit stop
        """
        return self.pit_stop_process(car_before, car_after, lap_time)

    def sample_pit_stop_duration(self, lap: int) -> float:
        """Gives the duration of the pit stop

        Returns:
            float: The time spent in the pit stop in milliseconds
        """
        return self.pit_stop_duration_process(lap)

    def write_info(self, f, lap_no):
        f.write(f'\n{self.race_id},{self.driver},{self.constructor},{self.course},{get_seconds_from_timedelta(self.current_time)},{self.year},{self.laps_since_pit_stop},{lap_no},{self.overtaking_mode},{self.pit_stopping},{self.pit_stop_duration},{get_seconds_from_timedelta(self.sampled_lap_time)}')