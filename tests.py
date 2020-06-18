import numpy as np
from scipy.optimize import minimize
from math import e, pi, log
import matplotlib.pyplot as plt


f = lambda p: (e/pi) ** p 
df = lambda p: ((e/pi) ** p) * (1 - log(pi))

def func(params):
    x = params[0]
    return f(x)

'''
x = [i/10 for i in range(-150, 150)]
y = [f(p) for p in x]
dy = [df(p) for p in x]

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Profit Function')
ax1.plot(x, y)
ax1.set(ylabel='Value')
ax2.plot(x, dy)
ax2.set(ylabel='Value')
plt.show()
'''

print(round(.94343))