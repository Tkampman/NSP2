import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load your data from the CSV file
data = pd.read_csv("Lifetime groep A 30-11.csv")

# Assuming your data has two columns, 'x0000' and 'y0000'
x = data['Time [us] - histogram']
y = data['Counts - histogram']

# Define the multi-modal function, which is a sum of Gaussians
def func(t, tau, N0, B):
    N = N0 * np.exp(-t/tau) + B
    return N

# Initial guesses for the parameters (tau, N0, B)
initial_params = [1, 2.2, 0]  # Adjust as needed

#error list
weights = np.full(len(x), 0.5)

# Perform the curve fit
params, covariance = curve_fit(func, x, y, p0=initial_params, sigma=1/weights)
# Extract the standard deviations (sqrt of diagonal elements of the covariance matrix)
perr = np.sqrt(np.diag(covariance))

# Create a fitted curve using the parameters
fitted_curve = func(x, *params)

# Plot the original data and the fitted curve
# plt.errorbar(x, y, fmt='o', label="Data with Error", color='b')
plt.plot(x,y, '-', label="Leeftijd muon", color="blue")
plt.plot(x, fitted_curve, label="Fitted Curve", color='crimson')
plt.xlabel('Tijd (us)')
plt.ylabel('Aantal')
plt.title("Leeftijd distributie muon")
plt.xlim([0,16])
plt.ylim([0,50])

# Print the parameters of the fitted peaks and their errors
tau, N0, B = params
tau_error = perr[0]
print("Fitted Parameters (tau, N_0, B) with Errors:")
print(f"tau: {tau} ± {perr[0]}, N_0: {N0} ± {perr[1]}, B: {B} ± {perr[2]}")
plt.legend()
# plt.savefig('Lifetime data met fit.png', dpi=600)
plt.show()