# Modelling N-Player Sports with Gaussian Processes

Project for Machine Learning and the Physical World, part of the ACS and MLMI courses at the University of Cambridge.

The idea of this project is to predict the outcome of horse races by modeling the finishing times of horses in each race.

The general idea behind modeling a sport like this is to arrive at a probability distribution over N players, which describes the winning probability of each of the players. However, modeling the outcome of horse races is difficult because of 2 major obstacles:
1. There are a variable number of horses in each race. This number usually ranges from around 3-20.
2. There is often comparatively little data on each horse, and sometimes none. We have to find a way to factor in horses we have never seen run before into our model.

The proposed method works something like the following:
1. For each horse, we use a Gaussian process to estimate a function that outputs a mean and variance for the finishing time of a horse given a combination of input variables including things like course and horse statistics (possibly with dimensionality reduction steps).
2. We use those mean and variance predictions to create a series of normal distributions, representing the finishing times for each horse.
3. We sample points from those distributions. For each sampling round we take the lowest sampled value to be the winner. We count up the number of times each horse wins and divide by the number of samples to reach an estimate of the distribution of win probabilities.
4. If a horse has no or very little data to go on, we need to find another method of coming to a good estimate of win probability. This will be a task in itself, but we are provided with the starting prices (that is, the market odds of a horse winning before the race starts) which could be used to estimate the finishing time distribution parameters directly.

Data: 
https://www.kaggle.com/gdaley/hkracing
