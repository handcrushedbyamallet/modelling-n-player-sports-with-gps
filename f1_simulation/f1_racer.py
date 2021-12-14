from __future__ import annotations

import lap_times
import pit_stopping
import overtaking
from f1_race_course import F1RaceCourse


class F1Racer:
    """The F1Racer class is the functioning heart of this simulation. It
    represents a single driver, constructor, car combination and stores the
    models for each of the different subproblems for the system. Upon
    initialisation it fits models to each subproblem. These systems are used to
    govern how the racer performs in the race.
    """
    def __init__(self, driver: str, constructor: str, course: str, starting_time: float):
        self.driver = driver
        self.constructor = constructor
        self.course = course
        self.current_time = starting_time
        self.laps_since_pit_stop = 0
        self.initialise_lap_time_params()
        self.initialise_overtake_params()
        self.initialise_pit_stop_params()

    def initialise_lap_time_params(self):
        """Fits the model that will govern the lap times of the racer
        """
        self.lap_time_process = lap_times.make_lap_time_process(self.driver, self.constructor, self.course)

    def initialise_overtake_params(self):
        """Fits the model that will govern the racer's ability to overtake
        """
        self.overtake_process = overtaking.make_overtaking_process(self.driver, self.constructor, self.course)

    def initialise_pit_stop_params(self):
        """Fits the model that will govern the racer's need to pit stop as well
        as the duration of pit stops
        """
        self.pit_stop_process = pit_stopping.make_pit_stop_process(self.driver, self.constructor, self.course)
        self.pit_stop_duration_process = pit_stopping.make_pit_stop_duration_process(self.driver, self.constructor, self.course)

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

    
