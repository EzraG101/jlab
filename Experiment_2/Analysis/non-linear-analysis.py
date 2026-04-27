import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# from scipy.odr import ODR, Model, Data, RealData
from scipy.stats import chisquare

# Global Variables

DATA_DIR = ".\\data"
PLOT_DIR = ".\\plots"

ROOM_TEMPERATURE = 293 # (K)
BOLTZMANN = 1.3806503 * 10**-23 # (J/K)
ABSOLUTE_ZERO = -273.15 # (C)
R = 99.3 * 1000
Rerr = 0.01 * R

# Colors

# Helper Functions

def get_data(filename:str) -> dict:
    """
    Reads the csv file at filename and outputs its content as a dictionary
    whose keys represent headers and values are np.arrays of data 
    """
    # Load the data
    df = pd.read_csv(filename)
    column_names = list(df.columns)
    values = df.to_numpy()
    
    # Create the dictionary
    data_dict = {}
    index = 0
    for key in column_names:
        data_dict[key] = values[:, index]
        index += 1
    
    return data_dict

def measurements_to_mean_with_err(measurements:np.ndarray, N:int|None=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes an np.array of measurements and the number of repetition per
    measurement, N, and outputs an np.array (of length len(measurements)/N) 
    of the means of each measurement and an np.array of the standard errors
    of each measurement
    """
    # Default Behavior
    if N is None:
        N = len(measurements)

    # Check N and measurements consistency
    L = len(measurements)
    if len(measurements) % N != 0:
        raise ValueError("Length of Measurements array is not divisible by number of repetitions.")
    l = L // N

    # Create the arrays
    means = []
    errs = []
    for i in range(l):
        trials = measurements[N * i:N * (i + 1)]
        means.append(np.mean(trials))
        errs.append(np.std(trials, ddof=1) / np.sqrt(N))
    
    return np.array(means), np.array(errs)

def plot_y_vs_x(
        x:np.ndarray, 
        xerr:np.ndarray, 
        y:np.ndarray, 
        yerr:np.ndarray, 
        xlabel:str, 
        ylabel:str, 
        output_dir:str, 
        filename:str,
        close:bool=True,
        ) -> None:
    """
    Makes a plot of y versus x with errors yerr and xerr, and saves it.
    """
    ### THINK ABOUT UPDATE FOR 2D y
    # Required directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Make scatter plot with error bars
    fig, ax = plt.subplots()
    ax.errorbar(
        x=x, 
        y=y, 
        xerr=xerr, 
        yerr=yerr,
        fmt="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if close:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close('all')
    else:
        return fig, ax
    
def numerical_integral(
        x:np.ndarray, 
        xerr:np.ndarray, 
        y:np.ndarray, 
        yerr:np.ndarray
        ) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Takes in np.arrays of x and y values, along with their errors and computes the
    corresponding integral and the error on it
    """
    # Check consistency
    L = len(x)
    if np.size(x) != np.size(y, axis=0):
        raise ValueError("x and y input arrays should have compatible shapes.")
    elif np.shape(x) != np.shape(xerr):
        raise ValueError("x and xerr input arrays should have compatible shapes.")
    elif np.shape(y) != np.shape(yerr):
        raise ValueError("y and yerr input arrays should have compatible shapes.")
    
    # Sort arrays
    idx = np.argsort(x)
    x, xerr = x[idx], xerr[idx]
    y, yerr = y[idx], yerr[idx]
    
    # Compute integral with trapezoidal rule, keeping track of errors
    integral = 0
    err2 = 0
    
    for i in range(L-1):
        integral += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2 # trapezoid rule

        if i == 0: # left edge
            err2 += ((y[1] + y[0]) * xerr[0] / 2) ** 2 # x err contribution
            err2 += ((x[1] - x[0]) * yerr[0] / 2) ** 2 # y err contribution
        else: # middle
            err2 += ((y[i-1] - y[i+1]) * xerr[i] / 2) ** 2 # x err contribution
            err2 += ((x[i+1] - x[i-1]) * yerr[i] / 2) ** 2 # y err contribution
    
    # Right edge
    err2 += ((y[L-1] + y[L-2]) * xerr[L-1] / 2) ** 2 # x err contribution
    err2 += ((x[L-1] - x[L-2]) * yerr[L-1] / 2) ** 2 # y err contribution

    # Convert error squared to just error
    err = np.sqrt(err2)

    return integral, err

def weighted_linear(x, m, a):
    return m * (x - a)

def weighted_linear_odr(beta, x):
    return beta[0] * (x - beta[1])

def repeated_values_to_mean_with_err(x:np.ndarray, y:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Does what I want
    """
    temp_dict = {}

    for xval, yval in zip(x, y):
        if xval not in temp_dict:
            temp_dict[xval] = [yval]
        else:
            temp_dict[xval].append(yval)
    
    out_x = list(temp_dict.keys())
    out_y = []
    out_y_err = []
    for xval in out_x:
        ys = np.array(temp_dict[xval])
        out_y.append(np.mean(ys))
        out_y_err.append(np.std(ys, ddof=1)/np.sqrt(len(ys)))
    
    return np.array(out_x), np.array(out_y), np.array(out_y_err)

def find_G(R, C, f, ferr, g2, g2err):
    y = []
    for i in range(len(f)):
        y.append(g2[i] / (1 + (2 * np.pi * f[i] * C * R) ** 2))
    y = np.asarray(y)
    yerr = y * 0 # CHANGE TO TRUTH IF NECESSARY

    return numerical_integral(f, ferr, y, yerr)

def make_fit_func(f, ferr, g2, g2err):
    def fit_func(R, k, C):
        G = find_G(R, C, f, ferr, g2, g2err)[0]
        return 4 * ROOM_TEMPERATURE * R * k * G
    return fit_func

if __name__ == '__main__':

    calibration = get_data(DATA_DIR + "\\calibration-3_31.csv")
    f, g2, g2err = repeated_values_to_mean_with_err(calibration['f'], calibration['gain^2'])
    f = f * 1000
    ferr = f * 0.00001

    data = get_data(DATA_DIR + "\\data-3_31-fix.csv")
    Rs = measurements_to_mean_with_err(data['R'], 5)
    Rs = Rs[0] * 1000, Rs[1] * 1000 + Rs[0] * 10
    v2s = measurements_to_mean_with_err(data['V^2'], 5)
    v2s = v2s[0] * (10**-6), v2s[1] * (10**-6)

    fit_func = make_fit_func(f, ferr, g2, g2err)

    popt, pcov = curve_fit(fit_func, Rs[0], v2s[0], [BOLTZMANN, 79*(10**-12)], sigma=v2s[1], absolute_sigma=True)
    kfit, Cfit = popt
    perr = np.sqrt(np.diag(pcov))
    kerr, Cerr = perr

    xvals = np.linspace(min(Rs[0]), max(Rs[0]), 300)
    yvals = fit_func(xvals, kfit, Cfit)

    # chi2 = chisquare(v2s, fit_func(Rs[0], *popt), ddof=2)
    # print(chi2.statistic, chi2.pvalue)

    fig, ax = plot_y_vs_x(*Rs, *v2s, xlabel=r'Resistance [$\Omega$]', ylabel=r'$V^2$ [V$^2$]', output_dir=PLOT_DIR, filename="", close=False)
    ax.plot(xvals, yvals, label='fit')

    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "non-linear.png"), dpi=300)
    plt.close('all')

    print(f'k = {kfit*10**23:.3f}±{kerr*10**23:.3f}, C = {Cfit*10**12:.3f}±{Cerr*10**12:.3f}')