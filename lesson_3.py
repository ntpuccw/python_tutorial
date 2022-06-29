import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# for latex equations
# from IPython.display import Math, Latex
np.random.seed(1234)
norm.cdf([-1.0, 0, 1])
x = np.linspace(-5, 5, 100)
y = norm.pdf(x)
plt.plot(x, y, c="r", label="$\mu=0$", linewidth=3.0)
plt.plot(x + 1, y, c="b", label="$\mu=1$", linewidth=3.0)
plt.legend()
plt.show()

p = np.array([1, -8, 16, -2, 8])
x = np.linspace(-1, 5, 100)
# y=[np.polyval(p,i) for i in x]
y = np.polyval(p, x)
plt.plot(x, y)
plt.grid()
plt.show()

a = np.random.randn(100, 1)
plt.plot(a)
plt.show()


# %%
import sys
print(sys.path)
# %%
