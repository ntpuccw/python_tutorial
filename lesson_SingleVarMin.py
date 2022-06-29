# min f(x); x is a single variable
# %%
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# Examples are from Chapter 8
p = np.array([1, -8, 16, -2, 8])
f = lambda x : np.polyval(p, x)
x = np.linspace(-1, 5, 100)
y = f(x)
# y = np.polyval(p, x)
plt.plot(x, y)
plt.grid()
plt.show()

# res = opt.minimize_scalar(f, bounds=(1, 5), method='bounded')
res = opt.minimize_scalar(f, (2, 3, 5)) # the bracket (2,3,5) means f(3)<f(2)<f(5)
print(res)

# %% Example 8.2
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

f = lambda x : np.arctan(5/x) + np.arctan(2/(3-x))
x = np.linspace(0, 3, 1000)
y = f(x)
plt.plot(x,y)
plt.grid(True)
plt.show()
res = opt.minimize_scalar(f, bounds=(0.1, 2), method='bounded')
# res = opt.minimize_scalar(f, (0.1, 0.2, 1.5)) # the bracket (2,3,5) means f(3)<f(2)<f(5)
print(res)

# %% Example 8.3
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

def f(x): # define a function which needs more lines to code
    t = (10*np.pi - 5*x)/(2*np.pi)
    V = t**2 * np.sqrt(25 - t**2)/3
    return V

x = np.linspace(0, 6, 1000)
y = f(x)
plt.plot(x,y)
plt.grid(True)

g = lambda x : -f(x)
res = opt.minimize_scalar(g, bounds=(0.1, 2), method='bounded')
plt.text(res.x, -res.fun, 'X')
plt.show()
print(res)
# %%
