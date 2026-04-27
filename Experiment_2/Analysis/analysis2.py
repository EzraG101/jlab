import numpy as np

ks = np.array([1.524, 1.389, 1.486, 1.361, 1.011])
a = np.std(ks)/np.sqrt(5)
b = np.sqrt(0.015**2+0.014**2+0.010**2+0.012**2+0.004**2)/np.sqrt(5)
c = np.sqrt(a**2+b**2)

print(np.mean(ks), a, b, c, (1.3806503-np.mean(ks))/c)

Ts = np.array([-276.3]*3)
d = np.sqrt(3.3**2+3.3**2+3.2**2)/np.sqrt(3)

print(np.mean(Ts), np.std(Ts)/np.sqrt(3), d, (-273.15+276.3)/d)