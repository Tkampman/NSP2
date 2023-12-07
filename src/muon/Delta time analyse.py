import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from Lifetime_analyse import tau, tau_error

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

def distance_peak_func(peak1,peak2):
    midpoint = (peak1 + peak2) / 2
    distance = abs(midpoint - peak1)
    return distance

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
plt.legend()
# Print the parameters of the fitted peaks and their errors
A1, mu1, sigma1 = params1
A2, mu2, sigma2 = params2

print("Fitted Parameters (A, mu, sigma) for Each Peak:")
print(f"Peak 1:  A: {A1} ± {perr1[0]}, mu: {mu1} ± {perr1[1]}, sigma: {sigma1} ± {perr1[2]}")
print(f"Peak 2:  A: {A2} ± {perr2[0]}, mu: {mu2} ± {perr2[1]}, sigma: {sigma2} ± {perr2[2]}")

# Defining constants and parameters
# =============================================================================================
# Time of life in muon perspective
time = tau
error_time = tau_error
muon_mass = 1.883531627 * 10**(-28)
c = 299792458

# Parameters of the detector
detector_distance = 2.45  # in meters
detector_distance_error = 0.05 # in meters
detector_resolution = 0.5 # in nanoseconds
error_dist = (detector_distance_error/detector_distance)**2

# Parameters of velocity
distance_to_peak = distance_peak_func(mu1,mu2) # in nanoseconds
speed = abs(detector_distance/(distance_to_peak * 10**(-9)))
velocity_light = speed/c
error_deltatime = (detector_resolution/distance_to_peak)**2
error_velocity = speed * np.sqrt(error_dist + error_deltatime)

# Parameters of the gamma factor
gamma = 1/(np.sqrt(1 - ((speed)**2)/(c**2)))
error_gamma = (6.242 * 10**(9) * muon_mass*c**2 * (speed * error_velocity/(c**2 * (1 - (speed**2)/(c**2))**(3/2))))

# Parameters of distance traveled
distance_traveled = time * (10**(-6)) * speed
error_distance_traveled = distance_traveled * np.sqrt((error_velocity/speed)**2 + (error_time/time)**2)
distance_traveled_lab = distance_traveled * gamma
error_distance_traveled_lab = distance_traveled * gamma * np.sqrt((error_gamma / gamma) ** 2 + (error_distance_traveled / distance_traveled) ** 2)
print(f"velocity is {speed}")
print(f'Vergeleken met de snelheid van licht is dit: {round(velocity_light,3)} ± {round(error_velocity/c, 3)} c')

# Parameters of the energy 
energy_loss = distance_traveled * gamma * 0.2 * 0.001293 #GeV
error_energy_loss = energy_loss * np.sqrt((error_distance_traveled / distance_traveled)**2 + (error_gamma / gamma)**2) #GeV
energy_end = muon_mass * c**2 * gamma * 6.242 * 10**(9) #GeV
error_energy_end = muon_mass * c**2 * error_gamma * 6.242 * 10**(9) #GeV
energy_GeV = energy_loss + energy_end #GeV
energy_GeV_error = np.sqrt((error_energy_loss)**2 + (error_energy_end)**2) #GeV

# Relativistic effects
delta_time = time * gamma
# delta_time_error = np.sqrt(((speed**2 * time**2 * error_velocity**2)/(c**4 - c**2 * speed**2)) + ((1 - (speed/c)**2) * error_time**2))
delta_time_error_2 = delta_time * np.sqrt(((error_gamma)/(gamma))** 2 + ((tau_error)/(time))**2)

print(f"velocity is {speed}")
print(f'Vergeleken met de snelheid van licht is dit: {round(velocity_light,3)} ± {round(error_velocity/c, 3)} c')
print(f"De muon is ontstaan op een hoogte van : {distance_traveled} ± {error_distance_traveled} m in de ref frame van de muon")
print(f"De muon is ontstaan op een hoogte van : {distance_traveled_lab} ± {error_distance_traveled_lab} m in de ref frame van ons")
print(f"Gamma factor: {gamma}")
print(f'De levensduur van de muon in ons referentiestelsel is: {delta_time} ± {delta_time_error_2} us')
print(f"Mu recentered {distance_to_peak}")
print(f"De energie gemeten {energy_end} ± {error_energy_end}")
print(f"Het energie verlies is {energy_loss} ± {error_energy_loss}")
print(f"De gemiddelde begin energie is : {round(energy_GeV, 3)} ± {round(energy_GeV_error, 3)} GeV")

# plt.savefig('Delta time two peaks.png', dpi=600)
plt.show()

