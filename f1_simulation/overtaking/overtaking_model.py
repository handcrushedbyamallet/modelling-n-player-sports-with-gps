import random
from typing import Callable
import GPy
from dataprocessing import F1Dataset
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import create_overtaking_dataset
from create_overtaking_dataset import make_overtakes_dataset
import datetime

GPy.plotting.change_plotting_library('matplotlib')

data = F1Dataset('data')

def process_data(driver:str):
    ## Assume this won't error due to update step
    driverId = get_driver_id(driver) 
    lp = make_overtakes_dataset()
    
    lp = lp.join(data.races.set_index(['raceId'])[['year', 'circuitId']], on=['raceId'])
    lp = lp.join(data.results.set_index(['raceId', 'driverId'])[['constructorId']], on=['raceId', 'driverId'])
    lp = lp.join(data.qualifying.set_index(['raceId', 'driverId'])[['q1','q2','q3']], on=['raceId','driverId'], lsuffix="qual_pos")
    ## Take first qualifying time as number is variable
    lp['q1'] = pd.to_datetime(lp['q1'], format='%M:%S.%f', errors='coerce')
    lp['qualtime'] = list(map(lambda x: (x - datetime.datetime(1900, 1, 1)).total_seconds(), lp['q1']))
    
    try:
        lp = lp.loc[lp['driverId'] == driverId]
    except:
        lp = lp
        print("Driver not found")

    ## Missed overtakes
    lp['num_overtakes'] = lp.groupby(['raceId', 'driverId'])['position_change'].transform(lambda x: x[x > 0].sum())
    lp['stuck_behind'] = lp.groupby(['raceId', 'driverId'])['stuck_behind_driver'].transform(lambda x: x[x == True].sum())
    print("after:",lp)

    ## Calculate percentage of successful overtakes
    lp['success_perc'] = (lp['num_overtakes']/(lp['stuck_behind']+lp['num_overtakes']))
    overtaking = lp.groupby(['raceId', 'driverId']).first().reset_index()

    return overtaking


def make_overtaking_process(lap_time: int, driver: str, constructor: int, courseId: int, year: int):
    
    overtaking = process_data(driver)   
    overtaking.dropna(inplace=True)
    X = overtaking[['qualtime', 'year', 'circuitId', 'constructorId']]
    Y = overtaking[['success_perc']]
    
    print("initialise kernel")
    kernel = GPy.kern.RBF(input_dim=3, lengthscale=10)

    print("fit model")
    m = GPy.models.GPRegression(X,Y,kernel)
    print("done")
    m.optimize(messages=True)
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


# overtaking = process_data("hamilton")   
# make_overtaking_process(80, "hamilton", 1, 1, 2021)