import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("4-07\\data-calibration.csv")

data = np.array(df.values)

freq = data[:, 0]
g2 = data[:, 3]


G = np.trapezoid(g2, freq)

fig, ax = plt.subplots()
ax.plot(freq, g2, 'o-', color='tab:red')
ax.set_xlabel("Frequency [kHz]")
ax.set_ylabel("Gain^2")
ax.grid(False)
plt.tight_layout()
plt.savefig('gain-curve.png', dpi=300)
plt.close('all')

print(G*10**3)