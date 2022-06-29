# Show the graphs of various probability dirtributions
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# dir(norm()) see the inside of norm
# define constants
mu, sigma = 0, 1 
typeIErr = 0.05
cf = norm.ppf([typeIErr/2, 1-typeIErr/2])
xlim = [-5, 5]
x = np.arange(cf[0], cf[1], 0.001) # range of x in spec
x_all = np.arange(xlim[0], xlim[1], 0.001) # entire range of x, both in and out of spec
y = norm.pdf(x,0,1)
y2 = norm.pdf(x_all,0,1)

# build the plot
fig, ax = plt.subplots(figsize=(9,6)) # (width=6.4,height=4.8)
plt.style.use('fivethirtyeight')
ax.plot(x_all,y2)

ax.fill_between(x, y, 0, alpha=0.3, color='b')
ax.fill_between(x_all, y2, 0, alpha=0.1)
ax.set_xlim(xlim)
ax.set_xlabel('X')
ax.set_yticklabels([])
ax.set_title('Standard Normal Distribution',  fontsize = 'small')
# plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
plt.grid(True, linestyle='--', which='major')
plt.show()

#%% T distribution with animation as run in terminal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

xlim = [-5, 5]
x = np.linspace(xlim[0], xlim[1], 1000)
y = norm.pdf(x,0,1)
plt.plot(x,y, color="red", label ='N(0,1)')
# df = 1
df = np.append(np.arange(0.1,1,0.1), np.arange(2,30,2))
plt.axis([xlim[0], xlim[1],0,0.5])
for i in df:
    y=t.pdf(x, i)
    plt.plot(x,y, lw=1, color='blue')
    # plt.pause(0.5)

plt.title('T distribution')
plt.legend()  
plt.show()

# %%
# chi2 distribution with animation as run in terminal
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

xlim = [0, 50]
x = np.linspace(xlim[0], xlim[1], 1000)

# df = 1
df = np.arange(4,20,2)
plt.axis([xlim[0], xlim[1],0,0.2])
for i in df:
    y=chi2.pdf(x, i)
    plt.plot(x,y, lw=1, color='blue')
    # plt.pause(0.5)

plt.title(r'$\chi^2$ Distribution')
plt.show()
# %% Beta distribution
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

# a, b = 8, 2
a = 9
b = np.arange(1, a)
x = np.linspace(0,1,100)
for i in b :
    y=beta.pdf(x, a, i)
    plt.plot(x,y, lw=1, color='blue')
    
plt.title(r'$\beta(a, b)$ distribution with a > b', fontsize = 'medium')    
plt.show()

b = 9
a = np.arange(1, b)
for i in a :
    y=beta.pdf(x, i, b)
    plt.plot(x,y, lw=1, color="blue")

plt.title(r'$\beta(a, b)$ distribution with a < b', fontsize = 'medium')        
plt.show()

a = np.arange(1, 9)
for i in a :
    y=beta.pdf(x, i, i)
    plt.plot(x,y, lw=1, color='blue')

plt.title(r'$\beta(a, b)$ distribution with a = b', fontsize = 'medium')        
plt.show()

# %% Binomial distribution
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

N, p = 20, 0.8
x = np.arange(0, N+1)
y = binom.pmf(x, N, p)
fig, ax = plt.subplots(3, 1)
# ax.plot(x, y, 'bo', ms=8) # ms: marker size
ax[0].vlines(x, 0, y, colors='b', lw=5, alpha=0.9)
# ax[1].bar(x, y)
ax[1].stem(x, y)
Y = binom.cdf(x, N, p)
ax[2].plot(x, Y, drawstyle='steps-pre')
plt.show()

# %% Poisson distribution
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

lam = 10
x = np.arange(0, poisson.ppf(0.99, lam)+1)
y = poisson.pmf(x, lam)
fig, ax = plt.subplots(2, 1)
# ax.plot(x, y, 'bo', ms=8) # ms: marker size
ax[0].vlines(x, 0, y, colors='b', lw=5, alpha=0.9)
# ax[1].bar(x, y)
# ax[1].stem(x, y)
Y = poisson.cdf(x, lam)
ax[1].plot(x, Y, drawstyle='steps-pre')
plt.show()

# %%
