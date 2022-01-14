import random
from typing import Callable
from dataprocessing import F1Dataset
import GPy
import numpy as np

data = F1Dataset('data')

# Need to get the circuit ID of the courses
MIN_SAMPLES_REQUIRED = 1
def make_pit_stop_process(driver_id: str, constructor_id: str, course_id: str, year: int) -> Callable[[], bool]:
    laps_since_last_pitstop = 0

    df = data.pit_stops.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race') \
        .merge(data.results.set_index('raceId'), on=['raceId','driverId']) \
        .join(data.circuits.set_index('circuitId'), on='circuitId', rsuffix='_circuit')
    if len(df[df["circuitId"] == course_id]) >= MIN_SAMPLES_REQUIRED:
        df = df[df["circuitId"] == course_id]
    if len(df[df["constructorId"] == constructor_id])>= MIN_SAMPLES_REQUIRED:
        df = df[df["constructorId"] == constructor_id]

    if len(df) == 0:
        print("ERROR: not enough data returning random number")
        return lambda: random.choice([True, False])
    df=df.sort_values(by=['raceId', 'driverId', 'lap'])
    df = df.assign(since_last_pitstop=df.groupby(['raceId', 'driverId'])['lap'].diff())
    df = df[df['since_last_pitstop'].notna()]
    X = df[['year']]
    Y = df[['since_last_pitstop']]

    kernel = GPy.kern.RBF(input_dim=1, lengthscale=500)+GPy.kern.Bias(input_dim=1)
    if len(X) == 0:
        print("ERROR: not enough data returning random number")
        return lambda: 5000
    m = GPy.models.GPRegression(X,Y,kernel)
    #m.optimize(messages=True)
    prediction = m.predict(np.array([[year]]))


    #print(prediction)
    def sample_next_pitstop_laps():
        return int(np.random.normal(loc=prediction[0], scale=prediction[1]))


    next_pitstop_laps = sample_next_pitstop_laps()

    current_lap = 0
    def is_pit_stop():
        nonlocal laps_since_last_pitstop, next_pitstop_laps, current_lap
        current_lap += 1
        if laps_since_last_pitstop == next_pitstop_laps:
            laps_since_last_pitstop = 0
            next_pitstop_laps = sample_next_pitstop_laps()
            return True
        else:
            laps_since_last_pitstop += 1
            return False
    return is_pit_stop


def make_pit_stop_duration_process(driver_id: str, constructor_id: str, course_id: str, year: int) -> Callable[[], float]:
    df = data.pit_stops.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race') \
        .merge(data.results.set_index('raceId'), on=['raceId','driverId'], suffixes=('','_results')) \
        .join(data.circuits.set_index('circuitId'), on='circuitId', rsuffix='_circuit')
    if len(df[df["circuitId"] == course_id]) >= MIN_SAMPLES_REQUIRED:
        df = df[df["circuitId"] == course_id]
    if len(df[df["constructorId"] == constructor_id])>= MIN_SAMPLES_REQUIRED:
        df = df[df["constructorId"] == constructor_id]

    #TODO: Add dimension lap
    X = df[["year"]]
    Y = df[["milliseconds"]]
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=500)+GPy.kern.Bias(input_dim=1)
    if len(X) == 0:
        print("ERROR: not enough data returning random number")
        return lambda: 5000
    m = GPy.models.GPRegression(X,Y,kernel)
    #m.optimize(messages=True)
    prediction = m.predict(np.array([[year]]))
    def callable():
        return np.random.normal(loc=prediction[0], scale=prediction[1])
    return callable




