import random
from typing import Callable
import GPy
from dataprocessing import F1Dataset
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
import create_overtaking_dataset
from create_overtaking_dataset import make_overtakes_dataset

GPy.plotting.change_plotting_library('matplotlib')

data = F1Dataset('data')

def process_data(driver:str):
    driverId = get_driver_id(driver) 
    lp = make_overtakes_dataset()

    
    # lp = data.lap_times
    # qual = data.qualifying
    # ##Filter out laps with pitstops
    lp["didPitStop"] = np.logical_not(lp.join(data.pit_stops.set_index(["raceId", "driverId", "lap"]), on=['raceId', 'driverId', 'lap'], rsuffix='pit_stops')["stop"].isna())
    lp = lp.loc[lp["didPitStop"] == False]
    lp = lp.join(data.races.set_index(['raceId'])[['year', 'circuitId']], on=['raceId'])
    lp = lp.join(data.results.set_index(['raceId', 'driverId'])[['constructorId']], on=['raceId', 'driverId'])
    lp['avg_laptime'] = lp.groupby(['raceId', 'driverId'])['milliseconds'].transform(np.mean)
    #lp['num_drivers'] = lp.groupby(['raceId'])['driverId'].nunique()
    try:
        lp = lp.loc[lp['driverId'] == driverId]
        
    except:
        lp = lp
        print("Driver not found")

    #print(lp)
    ## Missed overtakes
    lp['num_overtakes'] = lp.groupby(['raceId', 'driverId'])['position_change'].transform(lambda x: x[x > 0].sum())
    lp['stuck_behind'] = lp.groupby(['raceId', 'driverId'])['stuck_behind_driver'].transform(lambda x: x[x == True].sum())
    print("after:",lp)
    K = 5
    lp['success_perc'] = (lp['num_overtakes']/(lp['stuck_behind']+lp['num_overtakes']))-lp['position'].transform(lambda x: K/x )
   
    # ## Account for new/unseen drivers
    
    return lp


def make_overtaking_process(lap_time: int, driver: str, constructor: int, courseId: int, year: int):
    
    #print(overtaking) 

    X = overtaking[['avg_laptime', 'year', 'circuitId', 'constructorId']]
    Y = overtaking[['success_perc']]
    
    print("initialise kernel")
    kernel = GPy.kern.RBF(input_dim=4, lengthscale=1000)

    print("fit model")
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize(messages=False)
    m.optimize_restarts(num_restarts = 10)

    #print(overtaking)
    m.plot(fixed_inputs=[(1,year),(2,courseId),(3,constructor)], plot_data=True)
    plt.show(block=True) 

    ##TODO: Either change input params to take ID, or write helper methods to converst strings to IDs

    ## Pass in lap time maybe?
    result = m.posterior_samples_f(np.array([[lap_time,year, courseId, constructor]]), size=1)
    print(result)
    return result

def get_driver_id(driver):
    try:
        driver_data = data.drivers
        driver_id = driver_data.loc[driver_data['driverRef'] == driver]['driverId'].item()
    except KeyError as e:
        print(e)
        return None
    return driver_id

def get_driver_ref(driverId):
    driver_data = data.drivers
    driver_ref = (driver_data.loc[driver_data['driverId'] == driverId])['driverRef'].item()
    return driver_ref


overtaking = process_data("heidfeld")   
make_overtaking_process(10000, "hamilton", 1, 1, 2021)