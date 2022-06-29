from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 30
x = norm.rvs(size=n)
y = np.ones((1, n))
fig = plt.figure(figsize=(8, 3))
plt.scatter(x, y, s=50, alpha=0.5)
plt.vlines(0, 0.98, 1.02, color="r", label="Population Mean")
plt.vlines(x.mean(), 0.98, 10.2, color="g", label="Sample Mean")
plt.vlines(np.median(x), 0.98, 10.2, color="b", label="Sample Median")
plt.xlim((-3, 3))
plt.ylim((0.98, 1.02))
plt.xlabel("X")
plt.legend()
plt.title("A sample of {} points".format(n))
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 100
area = 2 * np.random.randint(50, size=n)
x1 = norm.rvs(size=n)
x2 = norm.rvs(loc=1, scale=2, size=n)
y = norm.rvs(loc=1, scale=0.002, size=n)

# fig = plt.figure(figsize=(8,3))
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(x1, y, s=area, alpha=0.5)
ax2.scatter(x2, y, s=area, alpha=0.5, color="r")
ax1.set_xlim((-6, 6))
ax1.set_ylim((0.98, 1.02))
ax2.set_xlim((-6, 6))
ax2.set_ylim((0.98, 1.02))
# plt.xlabel('X')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

n = 200
area = 2 * np.random.randint(50, size=n)
x1 = beta.rvs(6, 2, size=n)
x2 = beta.rvs(2, 6, size=n)
y = norm.rvs(loc=1, scale=0.002, size=n)
# y = 0.99 + 0.01 * np.random.randint(4, size = n)
# fig = plt.figure(figsize=(8,3))
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(x1, y, s=area, alpha=0.5)
ax2.scatter(x2, y, s=area, alpha=0.5, color="r")
ax1.set_xlim((0, 1))
ax1.set_ylim((0.98, 1.02))
ax2.set_xlim((0, 1))
ax2.set_ylim((0.98, 1.02))
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import zscore

n = 20
x1 = norm.rvs(loc=40, scale=1, size=n)
x2 = zscore(x1)
y = norm.rvs(loc=1, scale=0.002, size=n)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.scatter(x1, y, s=50)
ax2.scatter(x2, y, s=50, color="r")
ax1.set_ylim((0.98, 1.02))
ax2.set_ylim((0.98, 1.02))
ax1.set_yticklabels("")
ax2.set_yticklabels("")
plt.show()
# %%
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 200
x = norm.rvs(size=n)
y = norm.rvs(loc=1, scale=0.002, size=n)
plt.scatter(x, y, s=50)
plt.ylim((0.98, 1.02))
# %%
import numpy as np

s = np.arange(5) + 1
x = 1/ (10 ** s)
# %%
