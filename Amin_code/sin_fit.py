import numpy as np
from scipy import optimize
from scipy.stats import norm

rng = np.random.RandomState(1)

def sinfit(data, a):
  x = np.array(data[0])

  y = a*np.sin(x)
  return y.ravel()

x = 10 * rng.rand(50)
y = 4*np.sin(x) + 0.1 * rng.randn(50)

popt, pcov = optimize.curve_fit(sinfit, (x,), y, p0=(2.,))

print(popt)
