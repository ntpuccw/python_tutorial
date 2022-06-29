# Supervised learning: Linear and Augmented Regression Model
# %%
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.linalg as LA

data_dir = '../Data/'
D = np.loadtxt(data_dir + 'la_2.txt', comments='%') # la_1.txt
# Use scatter
# figure, ax = plt.figure()
area = 30 + D[:,3] # the size of markers
# colors = D[:,2] # the colors of markers
# colors = ['red' if i == 0 else 'blue' for i in D[:,2]]
colors = [[1,0,0] if i == 0 else [0,0,1] for i in D[:,2]]
plt.scatter(D[:, 0], D[:, 1], \
    c = colors, s = area, \
        alpha = 0.5, marker = 'o' )
plt.grid(True)
plt.show()

# Plot(scatter) groups separately to include labels.
Idx = (D[:,2]==0)
plt.plot(D[Idx, 0], D[Idx, 1], 'ro',\
    alpha = 0.5, label = 'Group A')
# plt.plot(D[D[:,2]==0,0], D[D[:,2]==0,1],'bo')
Idx = (D[:,2]==1)
plt.plot(D[Idx,0], D[Idx,1],'bo', \
    alpha = 0.5, label = 'Group B')
plt.grid(True)
plt.legend()
# plt.legend(loc = 'upper right')

# Estimation of the Regression Model
n = len(D[:,0])
X = np.hstack((np.ones((n, 1)), D[:, 0:2]))
y = D[:,2]
b = LA.inv(X.T @ X) @ X.T @ y.T
# Draw regression line by discriminate function
# By y = f(x)
# x = np.array([0, 6])
# y = -(b[0]- 0.5 + b[1] * x) / b[2] # x2 = f(x1)
# plt.plot(x, y, lw = 3)
# By contour line of y = f(x1, x2)
f = lambda x : b[0] + b[1] * x[0] + b[2] * x[1]
x_min, x_max = plt.xlim() # get the x limits of the current axes
y_min, y_max = plt.ylim()
nx, ny = 100, 100
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
contours = plt.contour(X, Y, Z, [0.5],\
    colors = 'grey', linewidths = 2) # levels = [0.5, 0.51]
# plt.pcolormesh(X, Y, Z) # add rich background color
# plt.show()
# Draw regression line by posterior prob. function

#  Augmented regression model
x1 = D[:,0, np.newaxis] # n x 1
x2 = D[:,1, np.newaxis]
X = np.hstack((np.ones((n, 1)), x1, x2, x1*x2, x1**2, x2**2))
y = D[:,2, np.newaxis]
# b = LA.inv(X.T @ X) @ X.T @ y
b = LA.pinv(X) @ y
f = lambda x : b[0] + b[1] * x[0] + b[2] * x[1] +\
    b[3] * x[0]*x[1] + b[4] * x[0]**2 + b[5] * x[1]**2

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
contours = plt.contour(X, Y, Z, \
    [0.5], colors='g', linestyles = '--') # levels = [0.5, 0.51]
plt.show()

# %% Linear Model
from sklearn.linear_model import LinearRegression

data_dir = '../Data/'
D = np.loadtxt(data_dir + 'la_2.txt', comments='%')
n = len(D[:,0])
X = D[:, 0:2]
y = D[:,2]

Mdl = LinearRegression()
Mdl.fit(X, y)
y_hat = Mdl.predict(X)
R2 = Mdl.score(X, y) # R-square
intrcp = Mdl.intercept_
coeffs = Mdl.coef_


# %% This is tuple
x = (np.random.rand(5), np.random.rand(5) )
print(x)
print(np.shape(x))
# %% An example to show legend by colors
import pandas as pd
import matplotlib.pyplot as plt

penguins_data="https://raw.githubusercontent.com/datavizpyr/data/master/palmer_penguin_species.tsv"
# load penguns data with Pandas read_csv
df = pd.read_csv(penguins_data, sep="\t")
df = df.dropna()
df.head()

plt.figure(figsize=(8,6))
sp_names = ['Adelie', 'Gentoo', 'Chinstrap']
scatter = plt.scatter(df.culmen_length_mm, 
            df.culmen_depth_mm,
            s=150,
            c=df.species.astype('category').cat.codes)
plt.xlabel("Culmen Length", size=24)
plt.ylabel("Culmen Depth", size=24)
# add legend to the plot with names
plt.legend(handles=scatter.legend_elements()[0], 
           labels=sp_names,
           title="species")
plt.savefig("scatterplot_colored_by_variable_with_legend_matplotlib_Python.png",
                    format='png',dpi=150)
plt.show()                    
# %%
