import numpy as np

# measured parameters
counts_west = 49876
time_west = 175421 # in seconds
error_counts_west = np.sqrt(counts_west)

counts_east = 2300
time_east = 12666 # in seconds
error_counts_east = np.sqrt(counts_east)

# Calculate the rates
west_rate = counts_west / time_west
error_west_rate = error_counts_west / time_west

east_rate = counts_east / time_east
error_east_rate = error_counts_east / time_west

error_rate = np.sqrt((error_east_rate) ** 2 + (error_west_rate)** 2)

# Asymmetry formula
A = (west_rate - east_rate) / (west_rate + east_rate)
error_A = A * np.sqrt(((error_rate)/(west_rate - east_rate))**2 + ((error_rate)/(west_rate + east_rate))**2)

print(f"The asymmetry is {round(A * 100, 1)} ± {round(error_A * 100, 1)}%")
