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
    
    lp = data.lap_times
    ##Filter out laps with pitstops
    lp["didPitStop"] = np.logical_not(lp.join(data.pit_stops.set_index(["raceId", "driverId", "lap"]), on=['raceId', 'driverId', 'lap'], rsuffix='pit_stops')["stop"].isna())
    lp = lp.loc[lp["didPitStop"] == False]

    lp = lp.join(data.races.set_index(['raceId'])[['year']], on=['raceId'])
    lp = lp.join(data.results.set_index(['raceId', 'driverId'])[['constructorId']], on=['raceId', 'driverId'])
    
            
    lp['num_overtakes'] = lp.groupby(['raceId', 'driverId'])['position'].diff()
    lp['avg_laptime'] = lp.groupby(['raceId', 'driverId'])['milliseconds'].transform(np.mean)
    lp['overtakes'] = lp.groupby(['raceId', 'driverId'])['num_overtakes'].transform(lambda x: x[x > 0].sum())
    overtaking = lp.groupby(['raceId', 'driverId']).first().reset_index()

    
    overtaking = overtaking.loc[overtaking['raceId'] == 841]
    
    X = overtaking[['avg_laptime','driverId', 'year', 'constructorId']]
    Y = overtaking[['overtakes']]
    
 
    kernel = GPy.kern.RBF(input_dim=4, lengthscale=500)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize(messages=True)
    m.optimize_restarts(num_restarts = 10)
    # #print(m.predict([[10000], [841],[1]]))
    # # m.plot()
    # # plt.show(block=True) 

    # ## Taking slices to condition on second input param
    # slices = [841]
    # for i, y in zip(range(3), slices):
    #     m.optimize(messages=True)
    #     m.optimize_restarts(num_restarts = 10)
    
    ## Set course to 841 and driver to Hamilton
    m.plot(fixed_inputs=[(1,841), (2,2021), (3,1)], plot_data=True)
    plt.show(block=True) 

    ##TODO: Either change input params to take ID, or write helper methods to converst strings to IDs

    result = m.predict([[driver], [constructor], [course],[year]])
    print(result)

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