# Solve f(x) = 0 for x, the single variable
# %%
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x - np.exp(-x/2)
sol = opt.root(f, 1)
print(sol.x)

x = np.linspace(0, 3, 1000)
y = f(x)
plt.plot(x, y)
plt.text(sol.x, 0, 'O', color='r')
plt.grid(True)
plt.show()
# %% For polynomial function
import scipy.optimize as opt
import numpy as np

# method 1: find one at once
p = np.array([1, -3, 2]) # poly coefs from high to low
f = lambda x : np.polyval(p, x)
sol = opt.root(f, 3)
print(sol.x)

# method 2: fins all roots
r = np.roots(p)
print(r)
# %% Solve for a complicated function
# Example 7.2
import scipy.integrate as integral
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# g = lambda x : np.exp(-x**2/2)/(2*np.pi)
# f = lambda x : integral.quad(g, -np.inf, x) - 0.9
def f(x, prob):
    g = lambda x : np.exp(-x**2/2)/np.sqrt(2*np.pi)
    return integral.quad(g, -np.inf, x)[0] - prob

sol = opt.root(f, 1, args=0.9) # pass an argument 
print(sol.x)
xlim = [-5, 5]
x = np.linspace(xlim[0], xlim[1], 100)
vec_f = np.vectorize(f)
y = vec_f(x, 0.9)
plt.plot(x,y)
plt.grid(True)
plt.hlines(0, xlim[0], xlim[1], color = 'r')
plt.vlines(sol.x, min(y), max(y), color = 'g')
plt.show()
# %% Exercise 7: f(x) = betapdf(2,6) - betapdf(4,2) =0
import scipy.optimize as opt
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
a1, b1, a2, b2 = 2, 6, 4, 2
def f(x, a1, b1, a2, b2) :
    return beta.pdf(x, a1, b1) - beta.pdf(x, a2, b2)

sol = opt.root(f, 0.5, args=(a1, b1, a2, b2))
print(sol.x)


x =np.linspace(0, 1, 1000)
y =f(x,a1, b1, a2, b2 )
plt.plot(x, y, label = 'f(x)')
plt.grid(True)
plt.hlines(0, 0, 1, color = 'r')
plt.vlines(sol.x, min(y), max(y), color = 'g')
plt.plot(x, beta.pdf(x, a1, b1), color = 'b', label = 'betapdf({},{})'.format(a1, b1)) 
plt.plot(x, beta.pdf(x, a2, b2), color = 'b', linestyle = '--', label = 'betapdf({},{})'.format(a2, b2))
plt.legend()
plt.show()

# %%
