from concurrent.futures import process
from f1_racer import F1Racer

from typing import List

from timeit import default_timer
import numpy as np


def get_seconds_from_timedelta(time):
    return time / np.timedelta64(1, 's')

def simulate_lap(racers: List[F1Racer], lap_number: int) -> List[F1Racer]:
    """Simulates a lap of a Formula 1 race.

    Args:
        racers (List[F1Racer]): A list of racers at the start of the lap

    Returns:
        List[F1Racer]: A list of racers with updated parameters at the end of the lap
    """

    racers = sorted(racers, key=lambda x: x.current_time)

    for pos, racer in enumerate(racers):
        # For logging
        racer.overtaking_mode = None
        racer.pit_stopping = None
        racer.pit_stop_duration = None
        racer.sampled_lap_time = None

        ##### Sample lap time ##### 
        lap_time = racer.sample_lap_time(lap_number)
        racer.current_time += lap_time
        racer.sampled_lap_time = lap_time

        ##### Overtakes ##### 
        if pos != 0:  # first place can't get stuck behind another car
            # TODO: Model lapping dynamics? What happens when racers are a lap behind leaders?
            previous_racer = racers[pos-1]
            # TODO Fix this so a car can overtake multiple other cars in one lap
            if racer.current_time < previous_racer.current_time:
                overtakes = racer.sample_overtake(get_seconds_from_timedelta(lap_time))
                if not overtakes:
                    racer.current_time = previous_racer.current_time  # Cap the lap time 
                    racer.overtaking_mode = 'stuck'
                else:
                    racer.overtaking_mode = 'success'
            else:
                racer.overtaking_mode = None

        ###### Pit stopping ###### 
        # We ignore relationships between drivers when they are pit stopping
        if racer.sample_pit_stop():
            pit_stop_time = racer.sample_pit_stop_duration()
            racer.current_time += np.timedelta64(int(pit_stop_time), 'ms')
            racer.laps_since_pit_stop = 0
            racer.pit_stopping = True
            racer.pit_stop_duration = pit_stop_time / 1000  # Convert to seconds
        else:
            racer.laps_since_pit_stop += 1
            racer.pit_stopping = False
            racer.pit_stop_duration = None

    racers = sorted(racers, key=lambda x: x.current_time)  # Sort again for good measure
    # print("#"*40 + "Lap Finished" + "#"*40)
    # print(f"Lap Times: {lap_times}, Pit Stops: {pit_stop_times}, Num Pit Stops: {num_pit_stops} Overtakes {num_overtakes}")
    with open('results.csv', 'a') as f:
        for racer in racers:
            racer.write_info(f, lap_number)
    return racers


def simulate_race(racers: List[F1Racer],
                  num_laps: int) -> List[F1Racer]:
    """Simulates an entire race from a list of F1 Racer objects over `num_laps`
    laps

    Args:
        racers (List[F1Racer]): A list of F1Racer objects at the start of the race
        num_laps (int): The number of laps that the race lasts for

    Returns:
        List[F1Racer]: A list of F1Racer objects at the end of the race
    """
    for lap in range(num_laps):
        racers = simulate_lap(racers, lap)

    return racers
