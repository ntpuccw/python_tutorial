# 1. This file demonstrate to import all modules from three pakcages
# That save the prefix and write codes as in MATLAB
# However, this is not recommended for many reasons
# 2. Load mat file 

from pylab import * #  pylab wraps NumPy, SciPy and matplotlib.
from scipy.io import loadmat, savemat # fyi

# matrix multiplication         #| % matrix multiplication
A = rand(3, 3)                  #| A = rand(3, 3);
A[0:2, 1] = 4                   #| A(1:2, 2) = 4;
I = A @ inv(A)                  #| I = A * inv(A);
I = A.dot(inv(A))               #|
print(np.round(I,2))
print(np.round(A @ A, 2)) # Lnear ALgebra
print(np.round(A * A, 2)) # elementwise

# vector manipulations          | % vector manipulations
t = linspace(0, 4, int32(1e3))  #| t = linspace(0, 4, 1e3);
y1 = cos(t/2) * exp(-t)         #| y1 = cos(t/2) .* exp(-t);
y2 = cos(t/2) * exp(-5*t)       #| y2 = cos(t/2) .* exp(-5*t);

# plotting                      | % plotting
figure()                        #| figure; hold on
plot(t, y1, label='Slow decay') #| plot(t, y1)
plot(t, y2, label='Fast decay') #| plot(t, y2)
legend(loc='best')              #| legend('Slow decay', 'Fast decay')
show()                          #|

#%% Load MAT data 
X = loadmat('mix.mat')
X.keys()
x = X['x']
y = X['y']
x.shape
y.shape
print([x[-1], y[-1]]) # -1: the last index
X['x_mean']
# %%
