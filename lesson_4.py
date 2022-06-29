# for Loop; anonymous function lambda
import numpy as np
import matplotlib.pyplot as plt

f = [1, 2, 3]
x = np.linspace(-3 * np.pi, 3 * np.pi, 200)
for i in f:  #  always start with a colon
    y = np.sin(i * x)  # blocks of code are identified by indentation
    plt.plot(x, y, label="sin({}x)".format(i))

plt.legend()
plt.ylim(-2, 2)
plt.grid(True)
plt.xlabel("X")
plt.show()


# in-line function
fun = lambda x: np.cos(x) * np.exp(-0.5 * x)
y = fun(x)
plt.plot(x, y)
plt.show()
