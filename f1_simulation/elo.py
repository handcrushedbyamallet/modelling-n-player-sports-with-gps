import pandas as pd
from dataprocessing import F1Dataset
import itertools
from tqdm import tqdm

data = F1Dataset('data')
results = data.results
results = results.join(data.races.set_index(['raceId']), on='raceId', rsuffix='_race')
results['date'] = pd.to_datetime(results['date'])
results = results.sort_values(by=['date', 'position'], ascending=True)

mean_elo = 1500
elo_width = 400
k_factor = 64

def update_elo(winner_elo, loser_elo, num_players):
    """
    From: https://www.kaggle.com/kplauritzen/elo-ratings-in-python
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo / num_players
    loser_elo -= change_in_elo / num_players
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    """
    From: https://www.kaggle.com/kplauritzen/elo-ratings-in-python
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

races = results.raceId.unique().tolist()
drivers = results.driverId.unique().tolist()

elos = {driver: mean_elo for driver in drivers}

results['finishingElo'] = 0
print("Calculating elo ratings")
for race in tqdm(races):
    race_df = results.loc[results['raceId'] == race]
    num_racers = len(race_df.driverId.unique())
    for driver_1, driver_2 in itertools.combinations(race_df.driverId.unique(), 2):
        driver_1_pos = race_df.loc[race_df['driverId'] == driver_1, 'position'].values[0]
        driver_2_pos = race_df.loc[race_df['driverId'] == driver_2, 'position'].values[0]

        winner = driver_1 if driver_1_pos < driver_2_pos else driver_2

        if winner == driver_1:
            driver_1_elo, driver_2_elo = update_elo(elos[driver_1], elos[driver_2], num_racers)
        else:
            driver_2_elo, driver_1_elo = update_elo(elos[driver_2], elos[driver_1], num_racers)

        elos[driver_1] = driver_1_elo
        elos[driver_2] = driver_2_elo

    results.loc[results['raceId'] == race, 'finishingElo'] = race_df['driverId'].map(elos)

results['startingElo'] = results.groupby(['driverId'])['finishingElo'].shift(1).fillna(1500)
results.to_csv('data/elo_ratings.csv', index=False)
