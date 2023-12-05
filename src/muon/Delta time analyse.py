import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from muon.Lifetime_analyse import tau
# Load your data from the CSV file
data_pos = pd.read_csv("Delta time groep A 30-11.csv")
data_neg = pd.read_csv("Delta time tot 27-11-2023 Groep B gestolen.csv")

def func(x, A, mu, sigma):
    N = A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return N

def fit_data(data, func, initial_params):

    x = data['Time [ns] - histogram']
    y = data['Counts - histogram']

    weights = np.full(len(x), 0.5)

    params, covariance = curve_fit(func, x, y, p0=initial_params, sigma=1/weights)

    perr = np.sqrt(np.diag(covariance))

    fitted_curve = func(x, *params)

    return params, perr, fitted_curve, x, y

params1, perr1, fitted_curve1, x1, y1 = fit_data(data_pos, func, [600, 7, 1])
params2, perr2, fitted_curve2, x2, y2 = fit_data(data_neg, func, [600, -10, 1])

# Plotting
plt.errorbar(x1, y1, fmt='o', color='seagreen', label='positieve tijd')
plt.plot(x1, fitted_curve1, label="Fit positief", color='limegreen')
plt.errorbar(x2, y2, fmt='o', color='mediumblue', label='negatieve tijd')
plt.plot(x2, fitted_curve2, label="Fit negatief", color='blue')
plt.xlim([-25, 25])
plt.xlabel('Tijd (ns)')
plt.ylabel('Aantal')
plt.title('Delta tijd meting')

# Print the parameters of the fitted peaks and their errors
A1, mu1, sigma1 = params1
A2, mu2, sigma2 = params2

print("Fitted Parameters (A, mu, sigma) for Each Peak:")
print(f"Peak 1:  A: {A1} ± {perr1[0]}, mu: {mu1} ± {perr1[1]}, sigma: {sigma2} ± {perr1[2]}")
print(f"Peak 2:  A: {A2} ± {perr2[0]}, mu: {mu1} ± {perr2[1]}, sigma: {sigma2} ± {perr2[2]}")


# Parameters of the detector
detector_distance = 2.45  # in meters
detector_distance_error = 0.05 # in meters
detector_resolution = 0.5 # in nanoseconds

# Calculating the midpoint between the two peaks using the mean of each peak
midpoint = (mu1 + mu2) / 2

# Calculating the distance in nanoseconds from the midpoint to each peak
distance_to_peak1 = abs(midpoint - mu1)
distance_to_peak2 = abs(midpoint - mu2)

# calculation of the velocity
c = 3 * 10**8
speed = abs(detector_distance/(distance_to_peak1 * 10**(-9)))

# error in velocity
error_dist = (detector_distance_error/detector_distance)**2
error_mu = (detector_resolution/distance_to_peak1)**2
error = speed * np.sqrt(error_dist + error_mu)

# Velocity compared to the speed of light
velocity_light = speed/c

print(f"velocity is {speed}")
print(f'Vergeleken met de snelheid van licht is dit: {round(velocity_light,3)} ± {round(error/c, 3)} c')

# Time of life in muon perspective
time = 2.0257199464354523
error_time = 0.034040854478832844

# Energy at creation
gamma = 1/(np.sqrt(1 - ((speed)**2)/(c**2)))
muon_mass = 1.883531627 * 10**(-28)
error_gamma = (6.242 * 10**(9) * muon_mass*c**2 * (speed * error/(c**2 * (1 - (speed**2)/(c**2))**(3/2))))


distance_traveled = time * (10**(-6)) * speed
error_distance_traveled = 0.2 * distance_traveled * np.sqrt((error/speed)**2 + (error_time/time))
# energy_loss = distance_traveled * gamma * loss factor (MeV/g/cm^2) in MeV
energy_loss = distance_traveled * gamma * 100 * 2 * 0.001293
error_energy_loss = energy_loss * np.sqrt((error_distance_traveled / distance_traveled)**2 + (error_gamma / gamma)**2) 
energy_J = energy_loss + muon_mass * c**2 * gamma
energy_GeV = energy_J * 6.242 * 10**(9)

energy_GeV_error = np.sqrt((0.2 * distance_traveled * np.sqrt((error/speed)**2 + 
(error_time/time)**2))**2 + (6.242 * 10**(9) * muon_mass*c**2 * (speed * error/(c**2 * (1 - (speed**2)/(c**2))**(3/2))))**2)

print(f"De muon is ontstaan op een hoogt van : {distance_traveled} ± {0.2 * distance_traveled * np.sqrt((error/speed)**2 + (error_time/time))} m in de ref frame van de muon")
print(f"De muon is ontstaan op een hoogt van : {distance_traveled * gamma} m in de ref frame van ons")
print(f"De gemiddelde energie is : {round(energy_GeV, 3)} ± {round(energy_GeV_error, 3)} GeV")
print(f"Gamma factor: {gamma}")

# Relativistic effects
delta_time = time / np.sqrt(1 - (speed/c)**2)
delta_time_error = np.sqrt(((speed**2 * time**2 * error**2)/(c**4 - c**2 * speed**2)) + ((1 - (speed/c)**2) * error_time**2))

print(f'De eigentijd van de muon is: {delta_time} ± {delta_time_error} us')
print(f"sigma/sqrt(N) {perr1[2]/np.sqrt(len(x1))}")
print(f"Mu recentered {distance_to_peak1}")
print(f"De energie gemeten {muon_mass * c**2 * gamma * 6.242 * 10**(9)} ± {(6.242 * 10**(9) * muon_mass*c**2 * (speed * error/(c**2 * (1 - (speed**2)/(c**2))**(3/2))))}")

print(f"Het energie verlies is {energy_loss} ± {error_energy_loss}")
plt.legend()
# plt.savefig('Delta time two peaks.png', dpi=600)
plt.show()

