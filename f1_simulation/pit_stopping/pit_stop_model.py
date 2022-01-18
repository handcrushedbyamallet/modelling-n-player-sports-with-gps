import random
from typing import Callable
from dataprocessing import F1Dataset
import GPy
import numpy as np

data = F1Dataset('data')

# Need to get the circuit ID of the courses
MIN_SAMPLES_REQUIRED = 1
MIN_SAMPLES_REQUIRED_PIT_DECISION = 100

models_created = {}

def make_pit_stop_process(driver_id: str, constructor_id: str, course_id: str, year: int) -> Callable[[float, float, float], bool]:
    def get_model(course_id: str, year: int):
        global models_created
        if (course_id, year) in models_created.keys():
            return models_created[(course_id, year)]
        lap_times_tmp = data.lap_times.sort_values(by=['raceId', 'driverId', 'lap'])
        lap_times = lap_times_tmp.assign(accumulated_time=lap_times_tmp.groupby(['raceId','driverId'])['milliseconds'].cumsum())

        pit_stops = data.pit_stops
        pit_stops = pit_stops.assign(last_stop=pit_stops['lap'])

        df = lap_times.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race') \
            .join(data.circuits.set_index('circuitId'), on='circuitId', rsuffix='_circuit') \
            .merge(pit_stops, on=['raceId', 'driverId', 'lap'], how='left', indicator = 'stop_indi')

        races_with_pit_stops = df['raceId'][~(df['last_stop'].isna())].unique()
        df = df.loc[df['raceId'].isin(races_with_pit_stops)]

        df['last_stop'] = df['last_stop'].ffill().fillna(0)
        df=df.assign(time_last_stop = (df['lap']-df['last_stop'])*df['milliseconds_x'])

        df['stop'] = np.where(df['stop_indi']=='both', 1, 0)

        if len(df[df["circuitId"] == course_id]) >= MIN_SAMPLES_REQUIRED_PIT_DECISION:
            df = df[df["circuitId"] == course_id]

        if len(df[df["year"] == year]) >= MIN_SAMPLES_REQUIRED_PIT_DECISION:
            df = df[df["year"] == year]


        df = df.sort_values(by=['raceId','accumulated_time'])
        df = df.assign(car_before=df['accumulated_time'].diff().fillna(1e9))
        df = df.assign(car_after=(-df['accumulated_time'].diff(periods=-1)).fillna(1e9))

        x_params = ["car_before", "car_after", "time_last_stop"]
        if len(df[df["year"] == year]) < MIN_SAMPLES_REQUIRED_PIT_DECISION:
            x_params += ['year']

        X = df[x_params]
        Y = df[['stop']]

        kernel = GPy.kern.RBF(input_dim=1, lengthscale=500)+GPy.kern.Bias(input_dim=len(x_params))
        if len(X) == 0:
            print("ERROR: not enough data returning random number")
            return lambda: False
        m = GPy.models.GPRegression(X,Y,kernel)
        m.optimize(messages=False)
        models_created[(course_id, year)] = m
        return m

    time_since_last_pitstop = 0
    m = get_model(course_id, year)

    def is_pit_stop(car_before: float, car_after: float, lap_time: float):
        nonlocal time_since_last_pitstop
        time_since_last_pitstop += lap_time
        mean = m.predict(np.array([[car_before, car_after, time_since_last_pitstop]]))[0]
        if np.random.rand()<mean:
            time_since_last_pitstop = 0
            return True
        else:
            return False

    return is_pit_stop


def make_pit_stop_duration_process(driver_id: str, constructor_id: str, course_id: str, year: int) -> Callable[[float], float]:

    df = data.pit_stops.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race') \
        .merge(data.results.set_index('raceId'), on=['raceId','driverId'], suffixes=('','_results')) \
        .join(data.circuits.set_index('circuitId'), on='circuitId', rsuffix='_circuit')

    if len(df[df["constructorId"] == constructor_id])>= MIN_SAMPLES_REQUIRED:
        df = df[df["constructorId"] == constructor_id]

    if len(df[df["year"] == year]) >= MIN_SAMPLES_REQUIRED:
        df = df[df["year"] == year]

    X = df[["lap"]]
    Y = df[["milliseconds"]]
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=500)+GPy.kern.Bias(input_dim=1)
    if len(X) == 0:
        print("ERROR: not enough data returning random number")
        return lambda: 5000
    m = GPy.models.GPRegression(X,Y,kernel)
    #m.optimize(messages=False)

    def callable(lap):
        prediction = m.predict(np.array([[lap]]))
        return np.random.normal(loc=prediction[0], scale=prediction[1])[0][0]

    return callable

