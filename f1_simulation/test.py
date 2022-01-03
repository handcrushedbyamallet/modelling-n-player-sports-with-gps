import pandas as pd
from f1_racer import F1Racer
from simulation import simulate_race
from dataprocessing import F1Dataset
import numpy as np

data = F1Dataset('data')

# Need to get the circuit ID of the courses
df = data.results.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race')
df = (df.set_index(['raceId', 'driverId'])
        .join(data.qualifying
                  .set_index(['raceId', 'driverId'])[['q1', 'q2', 'q3']]
                  .replace('\\N', np.nan)
                  .apply(pd.to_datetime, format='%M:%S.%f')
                  .min(axis=1)
                  .rename('top_quali')))  # Yikes

df['top_quali'] = df['top_quali'] - pd.to_datetime('1900-01-01', format='%Y-%m-%d')

df.reset_index(drop=False, inplace=True)
races = df['raceId'].unique()


for race_id in races:
    race = df.loc[df['raceId'] == race_id]

    assert len(race['circuitId'].unique()) == 1
    course_id = race['circuitId'].unique()[0]

    drivers = race['driverId'].tolist()
    constructors = race['constructorId'].tolist()
    year = data.races.loc[data.races['raceId'] == race_id, 'year'].values[0]
    num_laps = race['laps'].max()

    # TODO start the race in the correct order
    delay = 0
    racers = []
    for driver_id, constructor_id in zip(drivers, constructors):
        top_quali = race.loc[race['driverId'] == driver_id, 'top_quali'].values[0]
        racer = F1Racer(driver_id, constructor_id, course_id, year, starting_time=delay, total_laps=num_laps, top_quali=top_quali)
        delay += 20
        racers.append(racer)

    print([(racer.driver, racer.current_time) for racer in simulate_race(racers, num_laps)])
