import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files to check their format
data_pos_path = 'Delta time groep A 30-11.csv'
data_neg_path = 'Delta time tot 27-11-2023 Groep B gestolen.csv'

data_pos = pd.read_csv(data_pos_path)
data_neg = pd.read_csv(data_neg_path)

# Display the first few rows of each dataframe to understand their structure
data_pos_head = data_pos.head()
data_neg_head = data_neg.head()

data_pos_head, data_neg_head


# Corrected Cauchy distribution function
def func(z, gamma, z0):
    return 1 / (np.pi * gamma * (1 + ((z - z0) / gamma) ** 2))

# Function to fit data
def fit_data(data, func, initial_params):
    x = data['Time [ns] - histogram']
    y = data['Counts - histogram']

    # Assuming equal weights for simplicity, could be adjusted if needed
    weights = np.full(len(x), 0.5)

    params, covariance = curve_fit(func, x, y, p0=initial_params, sigma=1/weights)

    perr = np.sqrt(np.diag(covariance))

    fitted_curve = func(x, *params)

    return params, perr, fitted_curve, x, y

# Fit the data for both datasets
params1, perr1, fitted_curve1, x1, y1 = fit_data(data_pos, func, [10000, 10])
params2, perr2, fitted_curve2, x2, y2 = fit_data(data_neg, func, [10000, -10])

print(params1)
print(params2)

# Plotting
plt.errorbar(x1, y1, fmt='o', color='seagreen', label='Positieve tijd')
plt.plot(x1, fitted_curve1, label="Fit positief", color='limegreen')
plt.errorbar(x2, y2, fmt='o', color='mediumblue', label='Negatieve tijd')
plt.plot(x2, fitted_curve2, label="Fit negatief", color='blue')
plt.xlim([-25, 25])
plt.xlabel('Tijd (ns)')
plt.ylabel('Aantal')
plt.title('Delta tijd meting')
plt.legend()
plt.show()
