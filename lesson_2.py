import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))

plt.stem(x, y)
plt.show()

H = plt.stem(x, y, linefmt="grey", markerfmt="D", bottom=1.1, label="$e^{sin(x)}$")
H.markerline.set_markerfacecolor("green")
H.baseline.set_dashes([5, 2, 1, 2])
plt.xlabel("x - axis")
plt.ylabel("y - axis")
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.title(r"$e^{\sin{x}}$")
# plt.title('My first graph!')
plt.legend()
plt.show()
