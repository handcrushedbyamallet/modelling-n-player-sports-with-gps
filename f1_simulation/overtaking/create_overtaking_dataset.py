import pandas as pd
from dataprocessing import F1Dataset

def make_overtakes_dataset(cutoff_milliseconds=1000):
    data = F1Dataset('data')
    lap_times = data.lap_times

    lap_times['position_change'] = lap_times.groupby(['raceId', 'driverId'])['position'].diff().fillna(0)
    lap_times['total_time'] = lap_times.groupby(['raceId', 'driverId'])['milliseconds'].cumsum()

    lap_times = lap_times.sort_values(by=['raceId', 'lap', 'total_time'])
    lap_times['distance_to_leading_racer'] = lap_times.groupby(['raceId', 'lap'])['total_time'].diff()

    lap_times['close_to_leading_racer'] = lap_times['distance_to_leading_racer'] < cutoff_milliseconds
    lap_times['stuck_behind_driver'] = lap_times['close_to_leading_racer'] & (lap_times['position_change'] == 0)

    return lap_times
