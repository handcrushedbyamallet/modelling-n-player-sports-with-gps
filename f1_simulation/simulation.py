from f1_racer import F1Racer

from typing import List

from timeit import default_timer

def simulate_lap(racers: List[F1Racer]) -> List[F1Racer]:
    """Simulates a lap of a Formula 1 race.

    Args:
        racers (List[F1Racer]): A list of racers at the start of the lap

    Returns:
        List[F1Racer]: A list of racers with updated parameters at the end of the lap
    """

    overtakes = 0
    pit_stops = 0
    lap_times = 0
    racers = sorted(racers, key=lambda x: x.current_time)

    for pos, racer in enumerate(racers):
        ##### Sample lap time ##### 
        before = default_timer()
        racer.current_time += racer.sample_lap_time()
        after = default_timer()
        lap_times = after - before

        ##### Overtakes ##### 
        if pos != 0:  # first place can't get stuck behind another car
            # TODO: Model lapping dynamics? What happens when racers are a lap behind leaders?
            previous_racer = racers[pos-1]
            # TODO Fix this so a car can overtake multiple other cars in one lap
            if racer.current_time < previous_racer.current_time:
                before = default_timer()
                overtakes = racer.sample_overtake(previous_racer)
                after = default_timer()
                overtakes += after - before
                if not overtakes:
                    racer.current_time = previous_racer.current_time  # Cap the lap time 

        ###### Pit stopping ###### 
        # We ignore relationships between drivers when they are pit stopping
        before =  default_timer()
        if racer.sample_pit_stop():
            pit_stop_time = racer.sample_pit_stop_duration()
            racer.current_time += pit_stop_time 
            racer.laps_since_pit_stop = 0
        else:
            racer.laps_since_pit_stop += 1

    after =  default_timer()
    pit_stops += after - before
    racers = sorted(racers, key=lambda x: x.current_time)  # Sort again for good measure
    print("#"*40 + "Lap Finished" + "#"*40)
    print(f"Lap Times: {lap_times}, Pit Stops: {pit_stops}, Overtakes {overtakes}")
    exit()
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
        racers = simulate_lap(racers)

    return racers
