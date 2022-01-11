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

def process_data(driver:str):
    #overtaking_weight = 
    lp = data.lap_times
    qual = data.qualifying
    ##Filter out laps with pitstops
    lp["didPitStop"] = np.logical_not(lp.join(data.pit_stops.set_index(["raceId", "driverId", "lap"]), on=['raceId', 'driverId', 'lap'], rsuffix='pit_stops')["stop"].isna())
    lp = lp.loc[lp["didPitStop"] == False]
    lp = lp.join(data.races.set_index(['raceId'])[['year', 'circuitId']], on=['raceId'])
    lp = lp.join(data.qualifying.set_index(['raceId', 'driverId'])[['position']], on=['raceId','driverId'], lsuffix="qual_pos")
    lp = lp.join(data.results.set_index(['raceId', 'driverId'])[['constructorId']], on=['raceId', 'driverId'])
    
    driverId = get_driver_id(driver) 


    ## Missed overtakes
    # lp['lap_ft'] = lp.groupby(['raceId','driverId'])['milliseconds'].cumsum()
    
    # lp['missed'] = lp.groupby(['raceId', 'lap'])['lap_ft'].diff()
    

    lp['num_overtakes'] = lp.groupby(['raceId'])['position'].diff()
    #Normalise laptimes
    lp['avg_laptime'] = lp.groupby(['raceId'])['milliseconds'].transform(np.mean)
    #lp['avg_laptime'] = [float(i)/sum(lp['avg_laptime']) for i in lp['avg_laptime']]
    #Only successful overtakes 

    K = 10
    lp['overtakes'] = lp.groupby(['raceId'])['num_overtakes'].transform(lambda x: x[x > 0].sum())
    
    ## Weight according to qualifying position
    lp['ability'] = lp['overtakes'].transform(lambda x: lp['overtakes'] + lp['position']*K/x)
   
    
    overtaking = lp.groupby(['raceId', 'driverId']).first().reset_index()

    ## Account for new/unseen drivers
    try:
        overtaking = overtaking.loc[overtaking['driverId'] == driverId]
        return overtaking
    except:
        return overtaking
    


def make_overtaking_process(lap_time: int, driver: str, constructor: int, year: int):
    overtaking = process_data("heidfeld")   
    overtaking.dropna(inplace=True)
    print(overtaking) 

    #overtaking = overtaking.loc[overtaking['raceId'] == 841]
    X = overtaking[['avg_laptime', 'year', 'constructorId']]
    Y = overtaking[['ability']]
    
 
    kernel = GPy.kern.RBF(input_dim=3, lengthscale=100)
    #kernel = GPy.kern.Linear(1)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.optimize(messages=False)
    m.optimize_restarts(num_restarts = 10)

    #print(overtaking)
    m.plot(fixed_inputs=[(1,year),(2,constructor)], plot_data=True)
    plt.show(block=True) 

    # actual_value =  overtaking.loc[overtaking['avg_laptime'] == 108116.800000]

    ##TODO: Either change input params to take ID, or write helper methods to converst strings to IDs

    ## Pass in lap time maybe?
    result = m.posterior_samples_f(np.array([[lap_time,year,constructor]]), size=1)
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

def plus(val):
    return val[val > 0].sum()
def minus(val):
    return val[val < 0].sum()


make_overtaking_process(10000, "hamilton", 1, 2021)