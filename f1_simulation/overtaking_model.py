import random
from typing import Callable
import GPy
from dataprocessing import F1Dataset
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing

GPy.plotting.change_plotting_library('matplotlib')

data = F1Dataset('data')

def make_overtaking_process(driver: str, constructor: str, course: str, year: int):
    #join_df = data.lap_times.join(data.pit_stops.set_index(["raceId", "driverId", "lap"]).reset_index(), rsuffix="pitstop")
    #data.lap_times["didPitStop"] = np.logical_not(data.lap_times.join(data.pit_stops.set_index(["raceId", "driverId", "lap"]).reset_index(), rsuffix="pitstop")["stops"].isna())

    ##TODO: Remove laps with pit stops
    lp = data.lap_times

    races = lp.groupby(["raceId"])

    overtaking = pd.DataFrame()

    for race, df_race in races:
        for driverId in list(set(df_race['driverId'])):
            driverRef = get_driver_ref(driverId)
            position_diff = df_race.loc[df_race['driverId'] == driverId]['position'].diff()
            pos_change = sum(x for x in position_diff if x > 0)
            neg_change = sum(x for x in position_diff if x < 0)
            #print(pos_change)
            avg_laptime = df_race.loc[df_race['driverId'] == driverId]['milliseconds'].mean()
            new_df = pd.DataFrame({'raceId': [race],'driver': [driverId], 'overtakes':[pos_change], 'overtaken':[neg_change], 'avg_laptime': [avg_laptime]})
            overtaking = pd.concat([overtaking, new_df])

    
    # lp['num_overtakes'] = lp.groupby(['raceId', 'driverId'])['position'].diff()
    # lp['average_lap_time'] = lp.groupby(['raceId', 'driverId'])['milliseconds'].transform(np.mean)
    # lp['overtakes'] = lp.groupby(['raceId', 'driverId']).transform(lambda x: sum(y for y in x if y > 0))
    # overtakes = lp.groupby(['raceId', 'driverId']).mean()
    
    #overtaking = overtaking.loc[overtaking['raceId'] == 841]
    ##TODO: replace overtaken with starting position, qualifying time etc etc
    X = overtaking[['avg_laptime', 'raceId', 'driver']]
    Y = overtaking[['overtakes']]
    
 
    kernel = GPy.kern.RBF(input_dim=3, lengthscale=500)
    m = GPy.models.GPRegression(X,Y,kernel)
    # m.optimize(messages=True)
    # m.optimize_restarts(num_restarts = 10)
    print(m.predict([[10000], [841],[1]]))
    # m.plot()
    # plt.show(block=True) 

    ## Taking slices to condition on second input param
    #slices = 841]
    #for i, y in zip(range(3), slices):
        # m.optimize(messages=True)
        # m.optimize_restarts(num_restarts = 10)
    
    ##Set course to 841 and driver to Hamilton
    #m.plot(fixed_inputs=[(1,841), (2,1)], plot_data=True)
    #plt.show(block=True) 

    return lambda opponent: random.random() > 0.5

def get_driver_id(driver):
    driver_data = data.drivers
    driver_id = driver_data.loc[driver_data['driverRef'] == driver]['driverId']
    return driver_id

def get_driver_ref(driverId):
    driver_data = data.drivers
    driver_ref = (driver_data.loc[driver_data['driverId'] == driverId])['driverRef'].item()
    return driver_ref

def plus(val):
    return val[val > 0].sum()
def minus(val):
    return val[val < 0].sum()
    
make_overtaking_process("vettel", "mercedes", "monaco", 1999)