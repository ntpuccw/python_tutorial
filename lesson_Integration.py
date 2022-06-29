# Numerical integration
# %% Example 5.2 (1)
import scipy.integrate as integral
import matplotlib.pyplot as plt
import numpy as np

f = lambda x : np.sin(x**2)
lb, ub = 0, 1
result = integral.quad(f, lb, ub)
print(result[0])

x = np.linspace(0, np.pi/2, 100)
y = f(x)
fig, ax = plt.subplots() # (width=6.4,height=4.8)
# plt.style.use('fivethirtyeight')
ax.plot(x, y)
x = np.linspace(lb, ub, 100)
y = f(x)
# fill color between y and 0
ax.fill_between(x, y, 0, alpha=0.3, color='b')
ax.set_xlabel('X')
# ax.set_yticklabels([])
ax.set_title('The area = {:.4f}'.format(result[0]))
# plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
plt.grid(True, linestyle='--', which='major')
plt.show()
# %% Example 5.2(2)
import scipy.integrate as integral
import matplotlib.pyplot as plt
import numpy as np

f = lambda x : np.sqrt(x**2 + 4*x + 12)
lb, ub = -2, 6
result = integral.quad(f, lb, ub)
print(result[0])

x = np.linspace(lb - 1, ub + 2, 100)
y = f(x)
fig, ax = plt.subplots()  # (width=6.4,height=4.8)
# plt.style.use('fivethirtyeight')

ax.plot(x, y)
x = np.linspace(lb, ub, 100)
y = f(x)
ax.fill_between(x, y, 0, alpha=0.3, color='b')
ax.set_xlabel('X')
# ax.set_yticklabels([])
ax.set_title('The area = {:.4f}'.format(result[0]))
# plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
plt.grid(True, linestyle='--', which='major')
plt.show()

# %% Example 5.3
import scipy.integrate as integral
import matplotlib.pyplot as plt
import numpy as np

n = 100
xlim = [-5, 5]
f = lambda x : np.exp(-x**2/2)/np.sqrt(2*np.pi)
x = np.linspace(xlim[0], xlim[1], n)
# Method 1:
# P = np.zeros(len(x))
# for i in range(n):
#     P[i] = integral.quad(f, -np.inf, x[i])[0]

# Method 2:
def cdf(x):
    return integral.quad(f, -np.inf, x)[0]

vec_P = np.vectorize(cdf) # vectorized version of the function f
P = vec_P(x) # evalaute all in x
plt.plot(x, P, drawstyle='steps-pre') # stairs plot
plt.show()

# %%
