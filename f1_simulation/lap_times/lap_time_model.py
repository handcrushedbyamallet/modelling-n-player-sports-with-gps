import numpy as np
from typing import Callable
from f1_simulation.dataprocessing import F1Dataset
import pandas as pd
import GPy


def make_lap_time_process(driver_id: str, year: int, total_laps: int) -> Callable[[int, int], float]:
    data = F1Dataset('data')
    races = data.races
    years_races = races.loc[races['year'] == year][['raceId', 'circuitId']]

    # load qualification data,obtain fastest quali time at each race for normalisation purposes
    qualis = data.qualifying
    qrs = qualis.merge(years_races, on='raceId')
    top_time_idx = qrs.groupby(['raceId'])['q3'].transform(min) == qrs['q3']
    top_times = qrs[top_time_idx][['q3', 'raceId']]
    top_times['q3'] = pd.to_datetime(top_times['q3'], format='%M:%S.%f') \
                      - pd.to_datetime('00:00.000000', format='%M:%S.%f')

    # load lap times, process time string to timedelta, obtain number of laps in each race
    lap_times = data.lap_times
    years_laps = lap_times.merge(years_races, on='raceId')[['milliseconds', 'raceId', 'driverId', 'lap']]
    years_laps = years_laps.assign(time=pd.to_timedelta(years_laps['milliseconds'], unit='ms'))
    rc_laps = years_laps.groupby(['raceId'])['lap'].transform(max) == years_laps['lap']
    race_laps = years_laps[rc_laps][['raceId', 'lap']].drop_duplicates(ignore_index=True)

    # load pit stops, filter other years, process time and increase lap number for lap time sanitisation
    pit_stops = data.pit_stops
    years_pits = pit_stops.merge(years_races, on='raceId').loc[:, ['milliseconds', 'raceId', 'driverId', 'lap']]
    years_pits = years_pits.assign(time=pd.to_timedelta(years_pits['milliseconds'], unit='ms'))
    years_pits['lap'] += 1

    # filter out rows with data of other drivers
    years_laps = years_laps.loc[years_laps['driverId'] == driver_id].loc[:, ['raceId', 'lap', 'time']]
    years_pits = years_pits.loc[years_pits['driverId'] == driver_id].loc[:, ['raceId', 'lap', 'time']]

    # subtract pitstop time from lap time to have lower impact on lap times
    pitted_laps = years_laps.merge(years_pits, on=['raceId', 'lap'], suffixes=['_laps', '_pits'])
    pitted_laps = pitted_laps.assign(time=pitted_laps['time_laps'] - pitted_laps['time_pits'])
    pitted_laps.set_index(['raceId', 'lap'], inplace=True)
    years_laps.set_index(['raceId', 'lap'], inplace=True)
    years_laps.update(pitted_laps)
    years_laps.reset_index(inplace=True)

    # transform lap idx to ratio of whole race for normalisation of progress across races
    normalised_laps = years_laps.merge(race_laps, on='raceId', suffixes=['_idx', '_n'])
    normalised_laps = normalised_laps.assign(lap_r=normalised_laps['lap_idx'] / normalised_laps['lap_n'])

    # transform lap time into ratio of original time to fastest quali time at the race for normalisation across races
    normalised_laps = normalised_laps.merge(top_times, on='raceId')
    normalised_laps = normalised_laps.assign(
        rel_time=normalised_laps['time']/normalised_laps['q3']).loc[:, ['raceId', 'lap_idx', 'lap_r', 'rel_time']]

    kernel = GPy.kern.RBF(input_dim=1)
    model = GPy.models.GPRegression(normalised_laps.loc[:, 'lap_r'].values.reshape([-1, 1]),
                                    normalised_laps.loc[:, 'rel_time'].values.reshape([-1, 1]), kernel)
    model.optimize(messages=False, max_f_eval=1000)
    return lambda lap, laps_since_pitstop: model.posterior_samples_f(np.array([[lap / total_laps]]), size=1)[0, 0, 0]
