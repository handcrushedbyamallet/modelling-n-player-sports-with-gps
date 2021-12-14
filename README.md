# Modelling N-Player Sports with Gaussian Processes

Project for Machine Learning and the Physical World, part of the ACS and MLMI courses at the University of Cambridge.

In this project, we will build a simulation of the sport of Formula 1. This sport presents many interesting difficulties to us as machine learning practitioners. A few of these are listed below: 
* There are multiple processes that constitute the speed of a driver in a race, each of these possibly operating at different fidelities.
* The interactions between drivers in a race must be accounted for.
* As there are a variable number of drivers, classic machine learning models that require a fixed input size won't immidiately work.
* There is sometimes very little data on a driver (at least in the data we will be using), but we require modelling every driver to accurately model the race, so our system must be able to account for drivers with little data.

The goal of this project is to simulate the sport of Formula 1. The model devised here simulates each race lap by lap, with 3 processes determining the time spent in each each lap. Each of these processes is a learned function of the data, and is specific to each driver, manufacturer course combination. These processes are:
1. The time spent driving the lap independent of other cars and pit stops etc.
2. Overtaking behaviour: does the car get stuck behind another car. If so, how can we effectively model the overtake dynamics on a given lap?
3. The dynamics of pit stops - how long do they last and how do they influence the time a race takes to finish?

The winner being the driver with the smallest time on the final lap.

The process iterates through each driver, incrememting their time by a combination of the models listed above. The idea is for this simple breakdown of the dynamics of Formula 1 to allow us to model a complex sport, with multiple components and interactions between drivers with relative ease. This system should be sufficient to navigate the difficulties in modelling this sport.

If we finish head of schedule, we could look into using this system to generate odds for each driver winning a race by running a Monte Carlo simulation over the created model of each game. This should allow us to generate odds under different conditions. For example, we could use these output odds to look into how a driver could change their pit stopping strategy to maximise their chance of winning.

Data: https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020


## Setup
Grab the data and extract the files into the `data` folder.

### Accessing the data
The data can be accessed either directly from the csv files, or by using the `F1Dataset` object in `dataprocessing.py`.

```py
from f1_simulation.dataprocessing import F1Dataset

data = F1Dataset('data')  # takes the data folder as an arg

data.results  # returns a pandas dataframe for the 'results.csv' file
```

### Run the simulation
Right now, the simulation can be run through `test.py`. Simply run `python test.py` to run the trial simulation.

If you want to use the simulation manuually, you can do so through the `simulate_race` function in `simulation.py`

```py
from f1_simulation.f1_race_course import F1RaceCourse
from f1_simulation.f1_racer import F1Racer
from f1_simulation.simulation import simulate_race
from f1_simulation.dataprocessing import F1Dataset

data = F1Dataset('data')

# Need to get the circuit ID of the courses
df = data.results.join(data.races.set_index('raceId'), on='raceId', rsuffix='_race')

races = df['raceId'].unique()

for race_id in races:
    race = df.loc[df['raceId'] == race_id]

    assert len(race['circuitId'].unique()) == 1
    course_id = race['circuitId'].unique()[0]

    drivers = race['driverId'].tolist()
    constructors = race['constructorId'].tolist()

    # TODO start the race in the correct order
    delay = 0
    racers = []
    for driver_id, constructor_id in zip(drivers, constructors):
        racer = F1Racer(driver_id, constructor_id, course_id, starting_time=delay)
        delay += 20
        racers.append(racer)

    num_laps = race['laps'].max()
    print([(racer.driver, racer.current_time) for racer in simulate_race(racers, num_laps)])
```