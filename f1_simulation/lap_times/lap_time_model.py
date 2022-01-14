import numpy as np
from typing import Callable, Tuple
from dataprocessing import F1Dataset
import pandas as pd
import GPy
import datetime

processed_laps = dict()
processed_pits = dict()


def get_or_load_data(year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # return data if it was already processed
    global processed_laps
    global processed_pits
    if year in processed_pits:
        return processed_laps[year], processed_pits[year]

    data = F1Dataset('data')
    races = data.races
    years_races = races.loc[races['year'] == year][['raceId', 'circuitId']]

    # load qualification data,obtain fastest quali time at each race for normalisation purposes
    qualis = data.qualifying
    qrs = qualis.merge(years_races, on='raceId')
    qrs = qrs.loc[~pd.isnull(qrs['q3'])]
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

    # subtract pitstop time from lap time to have lower impact on lap times
    pitted_laps = years_laps.merge(years_pits, on=['raceId', 'lap', 'driverId'], suffixes=['_laps', '_pits'])
    pitted_laps = pitted_laps.assign(time=pitted_laps['time_laps'] - pitted_laps['time_pits'])
    pitted_laps.set_index(['raceId', 'lap', 'driverId'], inplace=True)
    years_laps.set_index(['raceId', 'lap', 'driverId'], inplace=True)
    years_laps.update(pitted_laps)
    years_laps.reset_index(inplace=True)
    pitted_laps.reset_index(inplace=True)

    # transform lap idx to ratio of whole race for normalisation of progress across races
    normalised_laps = years_laps.merge(race_laps, on='raceId', suffixes=['_idx', '_n'])
    normalised_laps = normalised_laps.assign(lap_r=normalised_laps['lap_idx'] / normalised_laps['lap_n'])

    # transform lap time into ratio of original time to fastest quali time at the race for normalisation across races
    normalised_laps = normalised_laps.merge(top_times, on='raceId')
    normalised_laps = normalised_laps.assign(
        rel_time=normalised_laps['time']/normalised_laps['q3']).loc[
                      :, ['raceId', 'driverId', 'lap_idx', 'lap_r', 'lap_n', 'rel_time']]

    processed_laps[year] = normalised_laps
    processed_pits[year] = pitted_laps

    return normalised_laps, pitted_laps


def make_lap_time_process(
        driver_id: int,
        year: int,
        total_laps: int,
        top_quali: datetime.timedelta,
        normalise_pit_laps: bool = True,
) -> Callable[[int, int], float]:

    normalised_laps, years_pits = get_or_load_data(year)

    # filter out rows with data of other drivers
    normalised_laps = normalised_laps.loc[normalised_laps['driverId'] == driver_id]
    years_pits = years_pits.loc[years_pits['driverId'] == driver_id]

    # define function to create laps since pitstop column and create it
    def laps_since_pit(race_id, lap_idx, laps):
        last_pit = years_pits.loc[(years_pits['raceId'] == race_id) & (years_pits['lap'] <= lap_idx)]['lap'].max()
        if pd.isnull(last_pit):
            res = lap_idx
        else:
            res = lap_idx - last_pit
        if normalise_pit_laps:
            return res / laps
        else:
            return res

    normalised_laps = normalised_laps.assign(
        laps_since_pit=normalised_laps.apply(lambda row: laps_since_pit(
            row['raceId'], row['lap_idx'], row['lap_n']), axis=1))

    # create model and optimise
    kernel = GPy.kern.RBF(input_dim=2)
    model = GPy.models.GPRegression(normalised_laps.loc[:, ['lap_r', 'laps_since_pit']].values.reshape([-1, 2]),
                                    normalised_laps.loc[:, 'rel_time'].values.reshape([-1, 1]), kernel)
    model.optimize(messages=False)

    # return prediction
    if normalise_pit_laps:
        return lambda lap, laps_since_pitstop: model.posterior_samples_f(
            np.array([[lap, laps_since_pitstop]]) / total_laps, size=1)[0, 0, 0] * top_quali
    else:
        return lambda lap, laps_since_pitstop: model.posterior_samples_f(
            np.array([[lap / total_laps, laps_since_pitstop]]), size=1)[0, 0, 0] * top_quali


if __name__ == '__main__':
    f = make_lap_time_process(842, 2020, 60, datetime.timedelta(minutes=1.2))
    print(f(30, 15))
