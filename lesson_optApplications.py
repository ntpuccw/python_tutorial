# Normal mixture problems:
# 1. Generate simulated samples of normal mixture distribution.
# 2. Sovle the constraint objective function for parameters
# 3. Plot the fitted graph with the histogram of the original data.

# %% 
import scipy.optimize as opt
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

# Plot a normal mixture pdf
mu1 = 0
s1 = 1
mu2 = 2
s2 = 2
pi1 = 0.7
pi2 = 1- pi1
# fig, ax = plt.figure()
x = np.linspace(-5, 10, 1000)
f = lambda x : pi1 * norm.pdf(x, mu1, s1) + pi2 * norm.pdf(x, mu2, s2)
y = f(x)
y1 = norm.pdf(x, mu1, s1)
plt.plot(x, y1, color = 'r', label = 'Normal({},{})'.format(mu1, s1**2))
y2 = norm.pdf(x, mu2, s2)
plt.plot(x, y2, color = 'r', \
    linestyle = '--', label = 'Normal({},{})'.format(mu2, s2**2))
plt.plot(x,y, \
    label = '{:.2f}Normal({},{})+{:.2f}Normal({},{})'.format(pi1, mu1, s1**2, pi2, mu2, s2**2))
plt.legend()
plt.show()

# generate rnadom samples of size n
n = 1000
# n1 = int(n*pi1)
n1 = binom.rvs(n, pi1)
n2 = n - n1
x1 = norm.rvs(loc=mu1, scale=s1, size=n1) 
x2 = norm.rvs(loc=mu2, scale=s2, size=n2) 
xs = np.append(x1, x2)
plt.hist(xs, bins=10, alpha=0.6, color='b', \
    edgecolor='y',linewidth=1, \
    density= True )


# MLE estimates of parameters pi1, mu1, s1, mu2, s2

L = lambda x : - np.sum(np.log(x[0] * norm.pdf(xs, x[1], x[2]) + \
    (1 - x[0]) * norm.pdf(xs, x[3], x[4])))

bnds = [(0, 1), (None, None), (0, np.inf), (None, None), (0, np.inf)]
opts = dict(disp = True, maxiter=1e4)
res = opt.minimize(L, x0=[0.5, 0, 1, 1, 1], 
    # method = 'L-BFGS-B', #'Nelder-Mead', 'trust-constr'
    bounds = bnds,
    # constraints = cons,
    options = opts,
    tol = 1e-8,)

f_est = lambda x : res.x[0] * norm.pdf(x, res.x[1], res.x[2]) + (1 - res.x[0]) * norm.pdf(x, res.x[3], res.x[4])
y = f_est(x)
plt.plot(x, y, color = 'y', linewidth = 3, \
    label = '{:.2f}Normal({:.2f},{:.2f})+{:.2f}Normal({:.2f},{:.2f})'.format(res.x[0], res.x[1], res.x[2]**2, (1-res.x[0]), res.x[3], res.x[4]**2))

plt.legend(loc='best', frameon=False)
# plt.show()
print(res)
# %% Approach by sklearn.mixture.GaussianMixture
from sklearn.mixture import GaussianMixture

X = xs.reshape(-1,1) # make it an array, not a vector
gm = GaussianMixture(n_components=2, \
    weights_init = [0.5, 0.5],
    means_init = [[0.5], [1.5]],
    max_iter = round(1e3),
    tol = 1e-6,
    n_init = 10, # The number of initializations to perform. The best results are kept.
    random_state=0).fit(X)

print(gm.weights_)
print(gm.means_)
print(gm.covariances_)
print(gm.lower_bound_) # ower bound value on the log-likelihood

f_est = lambda x : gm.weights_[0] * norm.pdf(x, gm.means_[0,0], np.sqrt(gm.covariances_[0,0])) +\
     gm.weights_[1] * norm.pdf(x, gm.means_[1,0], np.sqrt(gm.covariances_[1,0]))
y = f_est(x)
plt.plot(x, y, color = 'g', \
    linestyle = '--', linewidth = 2, \
    label = 'By sklearn.mixture.GaussianMixture')

plt.legend(loc='best', frameon=False)
plt.show()
# %%
