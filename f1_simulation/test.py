import pandas as pd
from f1_racer import F1Racer
from simulation import simulate_race
from dataprocessing import F1Dataset

data = F1Dataset('data')

# Need to get the circuit ID of the courses
df = data.results.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race')

races = df['raceId'].unique()

for race_id in races:
    race = df.loc[df['raceId'] == race_id]

    assert len(race['circuitId'].unique()) == 1
    course_id = race['circuitId'].unique()[0]

    drivers = race['driverId'].tolist()
    constructors = race['constructorId'].tolist()

    # TODO start the race in the correct order
    delay = 0
    racers = []
    for driver_id, constructor_id in zip(drivers, constructors):
        racer = F1Racer(driver_id, constructor_id, course_id, starting_time=delay)
        delay += 20
        racers.append(racer)

    num_laps = race['laps'].max()
    print([(racer.driver, racer.current_time) for racer in simulate_race(racers, num_laps)])
