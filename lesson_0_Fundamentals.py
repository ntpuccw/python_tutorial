# Fundamentals
import numpy as np  # numpy is a math package

help(np.arange)  # get help for function

# Logical operations
a = 3
if a > 1 and a < 5:
    print("TRUE")
else:
    print("FALSE")

if a > 5 or a < 10:
    print("OUTSIDE")
else:
    print("INSIDE")

# Precision
print(np.spacing(1))  # := eps in MATLAB
print(1e20 + 1 - 1e20)  # out of precison


# I/O: read mat file

from scipy.io import loadmat, savemat

data_dir = "../Data/"
X = loadmat(data_dir + "mix.mat")
print(X.keys())  # check with the variables inside the mat file
x = X["x"]
y = X["y"]

# I/O: txt file
data = np.loadtxt(data_dir + "Iris.txt")
print(data)
# txt file with comments
D = np.loadtxt(data_dir + "la_1.txt", comments="%")
print(D[0:5, 3])
