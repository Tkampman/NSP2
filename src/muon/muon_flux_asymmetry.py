import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# measured parameters
counts_east = 826
time_east = 4411 # in seconds
counts_west = 49876
time_west = 175421 # in seconds

# Calculate the rates
west_rate = counts_west / time_west
east_rate = counts_east / time_east

# Asymmetry formula
A = (west_rate - east_rate) / (west_rate + east_rate)

print(f"The asymmetry is {round(A* 100, 1)}%")
