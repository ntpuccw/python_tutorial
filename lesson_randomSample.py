# Generate random sample of a normal distribution and
# show  histogram , bixplot, normal plot ...
# %% Normal distribution; 
# random numbers, histogram, boxplot, normal plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import cumfreq  # for ECDF

rng = np.random.default_rng() # a random number generator
# histogram
x = rng.normal(loc=0, scale=1, size=1000)
# x = norm.rvs(loc=0, scale=1, size=1000, random_state=rng)
plt.hist(x, bins=10, alpha=0.6, color='b', edgecolor='y',linewidth=1)
plt.show()

fig, ax = plt.subplots(1, 1)
x = norm.rvs(size=1000) # standard normal distribution
# with and without stepfilled (no space in between bars)
# histtype=[bar, barstacked, step, stepfilled], default is “bar”
ax.hist(x, density=True, histtype='bar',  color="r", edgecolor="black",lw=2, alpha=0.6, rwidth=0.9)
# ax.hist(x, density=True,  color="b", edgecolor="black",lw=2, alpha=0.6,)
plt.show()

# various boxplot graphs
x1 = norm.rvs(size=1000)
x2 = norm.rvs(loc=1, scale=1, size=1000)
data = [x1, x2]
plt.boxplot(data, notch=True, vert=True, labels=['X1','X2'])
plt.show()

x1 = norm.rvs(size=1000)
x2 = norm.rvs(loc=1, scale=1, size=1000)
data = [x1, x2]
boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
flierprops = dict(marker='o', markerfacecolor='green', markersize=8, linestyle='none')
labels = ['N(0,1)','N(1,1)']
plt.boxplot(data,  boxprops=boxprops, flierprops=flierprops, labels=labels)
plt.show()

# normal plot
# qqplot by Scipy
import scipy.stats as stats
# Note: use numpy's random number generation
x = np.random.normal(loc = 20, scale = 5, size=100)   
stats.probplot(x, dist="norm", plot=plt)
plt.show()

# ecdf graph
n = 100
sample = norm.rvs(size=n)
# Method 1:
num_bins =  100
res = cumfreq(sample, num_bins)
x = res.lowerlimit + \
    np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
# plt.bar(x, res.cumcount/n, width=res.binsize)
plt.plot(x, res.cumcount/n, drawstyle='steps-pre')
plt.title('The Empirical CDF with {} steps'.format(num_bins), fontsize = 'small')
plt.show()

# Method 2
x = np.sort(sample)
Y = np.arange(1, n + 1) / n
plt.plot(x, Y, drawstyle='steps-pre')
plt.title('The Empirical CDF with {} steps'.format(num_bins), fontsize = 'small')
plt.show()

# %%
