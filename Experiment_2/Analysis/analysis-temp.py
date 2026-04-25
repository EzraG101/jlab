import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Global Variables

DATA_DIR = ".\\data"
PLOT_DIR = ".\\plots"

ROOM_TEMPERATURE = 293 # (K)

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
    return m*(x - a)

# Run code

if __name__ == "__main__":
    calibration = get_data(DATA_DIR + "\\calibration-4_02.csv")
    freqs = measurements_to_mean_with_err(calibration["f"], 3)
    gain2s = measurements_to_mean_with_err(calibration["gain^2"], 3)
    data = get_data(DATA_DIR + "\\data-4_02-fix.csv")
    
    plot_y_vs_x(*freqs, *gain2s, output_dir=PLOT_DIR, filename="temp-calibration.png", xlabel="Frequency [kHz]", ylabel="Gain^2")
    
    g2, g2err = gain2s
    f, ferr = freqs
    f = f * 1000
    ferr = ferr * 1000
    y = []
    yerr2 = []
    C = 50 * 10 ** -12
    Cerr = 0
    for i in range(len(f)):
        row_y = []
        row_err = []
        row_y.append(g2[i] / (1 + (2 * np.pi * f[i] * C * 99300) ** 2))

        err2 = (g2err[i] / (1 + (2 * np.pi * f[i] * C * 99300) ** 2)) ** 2 # g2 err contribution

        err2 += (g2[i] / (1 + (2 * np.pi * f[i] * C * 99300) ** 2) ** 2 * 2 * f[i] * (2 * np.pi * C * 99300) ** 2 * ferr[i]) ** 2 # f err contribution
        err2 += (g2[i] / (1 + (2 * np.pi * f[i] * C * 99300) ** 2) ** 2 * 2 * C * (2 * np.pi * f[i] * 99300) ** 2 * Cerr) ** 2 # C err contribution
        row_err.append(err2)
        y.append(row_y)
        yerr2.append(row_err)
    y = np.array(y)
    yerr2 = np.array(yerr2)
    yerr = np.sqrt(yerr2)

    Gval, Gerr = numerical_integral(f, ferr, y, yerr)

    V2 = np.asarray(data["V^2"], dtype="float") * 10 ** -6

    quant = V2 / (4 * (99.3 * 1000) * Gval)
    quant_err = quant * Gerr / Gval
    temp = np.asarray(data["T"], dtype="float")

    popt, pcov = curve_fit(weighted_linear, temp, quant)
    perr = np.sqrt(np.diag(pcov))

    fig, ax = plot_y_vs_x(temp, temp*0, quant, quant_err, output_dir=PLOT_DIR, filename="main-plot.png", xlabel="Temperature [C]", ylabel=r"$V^2/4RG$ [J]", close=False)

    xvals = np.linspace(-280, max(temp), 500)
    yvals = weighted_linear(xvals, *popt)

    ax.plot(xvals, yvals)
    ax.vlines(-273.15, min(yvals), max(yvals))
    ax.hlines(0, min(xvals), max(xvals))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "main-plot.png"), dpi=300)
    plt.close('all')

    print(f"k = {popt[0]*10**23:.3f}±{perr[0]*10**23:.3f}, T0 = {popt[1]:.1f}±{perr[1]:.1f}")