from __future__ import annotations

import lap_times
import pit_stopping
import overtaking

import datetime

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
        driver_id: str, 
        constructor_id: str, 
        course_id: str, 
        year: int, 
        starting_time: float,
        total_laps: int,
        top_quali: datetime.timedelta
    ):
        self.driver = driver_id
        self.constructor = constructor_id
        self.course = course_id
        self.current_time = starting_time
        self.year = year
        self.laps_since_pit_stop = 0
        self.initialise_lap_time_params(driver_id, year, total_laps, top_quali)
        self.initialise_overtake_params()
        self.initialise_pit_stop_params()

    def initialise_lap_time_params(self, driver_id: int, year: int, total_laps: int, top_quali: datetime.timedelta, normalise_pit_laps: bool = True):
        """Fits the model that will govern the lap times of the racer
        """
        self.lap_time_process = lap_times.make_lap_time_process(self.driver, self.constructor, self.course, self.year)

    def initialise_overtake_params(self):
        """Fits the model that will govern the racer's ability to overtake
        """
        self.overtake_process = overtaking.make_overtaking_process(self.driver, self.constructor, self.course, self.year)

    def initialise_pit_stop_params(self):
        """Fits the model that will govern the racer's need to pit stop as well
        as the duration of pit stops
        """
        self.pit_stop_process = pit_stopping.make_pit_stop_process(self.driver, self.constructor, self.course, self.year)
        self.pit_stop_duration_process = pit_stopping.make_pit_stop_duration_process(self.driver, self.constructor, self.course, self.year)

    def sample_lap_time(self) -> float:
        """Sample a lap time from the racer's lap time model and the course

        Returns:
            float: The time taken to complete the lap in milliseconds.
        """
        return self.lap_time_process()

    def sample_overtake(self, opponent: F1Racer) -> bool:
        """Sample whether this racer can overtake the `opponent` racer

        Args:
            opponent (F1Racer): The racer who may be overtaken

        Returns:
            bool: Whether this racer successfully overtook the `opponent` racer
        """
        return self.overtake_process(opponent)

    def sample_pit_stop(self) -> bool:
        """Samples whether the racer will go in for a pit stop

        Returns:
            bool: Whether the car goes in for a pit stop
        """
        return self.pit_stop_process()

    def sample_pit_stop_duration(self) -> float:
        """Gives the duration of the pit stop

        Returns:
            float: The time spent in the pit stop in milliseconds
        """
        return self.pit_stop_duration_process()
