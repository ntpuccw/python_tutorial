# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
# ##########################################
# check the mapping method :
# https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html
# This set goes from red (small value ~0) to blue (large value ~1)
cdit = {'red': [(0, 1, 1), (1, 0.7, 0.7)], # small value tends to red
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]} # large value tends to blue
# cdict = {'red':   [(0.0,  0.0, 0.0),
#                    (0.5,  1.0, 1.0),
#                    (1.0,  1.0, 1.0)],

#          'green': [(0.0,  0.0, 0.0),
#                    (0.25, 0.0, 0.0),
#                    (0.75, 1.0, 1.0),
#                    (1.0,  1.0, 1.0)],

#          'blue':  [(0.0,  0.0, 0.0),
#                    (0.5,  0.0, 0.0),
#                    (1.0,  1.0, 1.0)]}          
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes', cdit)

plt.cm.register_cmap(cmap=cmap) # register
# ##########################################
data_dir = '../Data/'
D = np.loadtxt(data_dir + 'la_2.txt', comments='%') # la_1.txt

Idx = (D[:,2]==0)
area = 2 * np.random.randint(100, size = D[Idx, 0].size)
plt.scatter(D[Idx, 0], D[Idx, 1], \
    c = 'r', s = area, \
        alpha = 0.5, marker = 'o' )

# plt.plot(D[Idx, 0], D[Idx, 1], 'ro',\
#     alpha = 0.5, label = 'Group A')
# plt.plot(D[D[:,2]==0,0], D[D[:,2]==0,1],'bo')
Idx = (D[:,2]==1)
plt.scatter(D[Idx, 0], D[Idx, 1], \
    c = 'b', s = area, \
        alpha = 0.5, marker = 'o' )
# plt.plot(D[Idx,0], D[Idx,1],'bo', \
#     alpha = 0.5, label = 'Group B')
plt.grid(True)
# plt.legend()
X = D[:, 0:2]
y = D[:,2]

Lda = LinearDiscriminantAnalysis(tol = 1e-6)
Lda.fit(X, y)
intrcp = Lda.intercept_
coeffs = Lda.coef_
print(Lda.score(X, y))

nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_ = np.linspace(x_min, x_max, nx)
y_ = np.linspace(y_min, y_max, ny)
xx, yy = np.meshgrid(x_, y_)
# use discriminant function 
# f = lambda x : intrcp + coeffs[0, 0] * x[0] + coeffs[0, 1] * x[1]
# Z = f([xx, yy])
# use posterior prob. 
Z = Lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
# map Z (the posterior prob.) value to color
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',\
    norm=colors.Normalize(0., 1.),\
    shading='auto', zorder = 0)
contours = plt.contour(xx, yy, Z, [0.5],\
    colors = 'white') # levels = [0.5, 0.51]

# plt.show()

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# X = D[:, 0:2]
# y = D[:,2]
Qda = QuadraticDiscriminantAnalysis(tol = 1e-6, store_covariance = True)
Qda.fit(X, y)
Qda.means_
Qda.covariance_
Qda.rotations_

Z = Qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
contours = plt.contour(xx, yy, Z, [0.5],\
    colors = 'white') # levels = [0.5, 0.51]

plt.title('The Linear and Quadratic Discriminant Boundary')
plt.show()

# %%
# fit and predictions and ellipse
Qda = QuadraticDiscriminantAnalysis(tol = 1e-6, store_covariance = True)
Qda.fit(X, y)
y_hat = Qda.predict(X)

tp = (y == y_hat)  # True Positive
tp0, tp1 = tp[y == 0], tp[y == 1]
X0, X1 = X[y == 0], X[y == 1]
X0_tp, X0_fp = X0[tp0], X0[~tp0]
X1_tp, X1_fp = X1[tp1], X1[~tp1]

fig = plt.subplot(1, 1, 1)
# class 0
plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

# class 1: dots
plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

Z = Qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), 
                   shading='auto', zorder=0)
contours = plt.contour(xx, yy, Z, [0.5],\
    colors = 'k') # levels = [0.5, 0.51]
# The classs' means
plt.plot(Qda.means_[0][0], Qda.means_[0][1],
             'o', color='yellow', markersize=15, markeredgecolor='grey')
plt.plot(Qda.means_[1][0], Qda.means_[1][1],
             'o', color='yellow', markersize=15, markeredgecolor='grey')

# Plot ellipse for each class
from scipy import linalg
import matplotlib as mpl

mean1, mean2 = Qda.means_[0], Qda.means_[1] 
cov1, cov2 = Qda.covariance_[0], Qda.covariance_[1]

def plot_ellipse(fig, mean, cov, color) :
    w, v = linalg.eigh(cov) # w: eigenvalues, v: eigenvector (columns)
    u = v[0] / linalg.norm(v[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * w[0] ** 0.5, 2 * w[1] ** 0.5,
                    180 + angle, facecolor=color,
                    edgecolor='black', linewidth=2)
    ell.set_clip_box(fig.bbox)
    ell.set_alpha(0.2)
    fig.add_artist(ell)
# fig.set_xticks(())
# fig.set_yticks(())
plot_ellipse(fig, mean1, cov1, 'red')
plot_ellipse(fig, mean2, cov2, 'blue')
plt.title('Ellipses for Normal Distributions')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()

# %%
#Create training and test datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.20, random_state = 5)

# %%
# MVN and its random samples
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from Lib_GenData import mvn_multiclass_data # self-defined

x, y = np.mgrid[-3:3.6:.05, -3.2:3:.05]
pos = np.dstack((x, y)) # nx x ny x 2
mean = [0.5, -0.2]
# cov = [[2.0, 0.3], [0.3, 0.5]]
cov = [[1, 0], [0, 1]]

# set up a multivariate Normal Distribution
mvn = multivariate_normal(mean = mean , cov = cov)
fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.contourf(x, y, mvn.pdf(pos))
ax = plt.axes(projection = '3d')
ax.plot_surface(x, y, mvn.pdf(pos), color = 'r',   rstride=4, cstride=4, alpha =0.6,
    cmap='ocean') # cmap = plt.cm.bone    

plt.show()
# generate random numbers
D = mvn.rvs( size = 200)
plt.scatter(D[:, 0], D[:, 1], marker='.', 
        color='blue')

plt.show()

# def mvn_multiclass_data(mean, cov, n) :
#     X = np.array([])
#     y = np.array([])
#     grp_size = mean.shape[0]
#     grp = np.arange(0, grp_size)
#     Idx = np.arange(0, grp_size*2+1, 2)
#     for i in grp :
#         mvn = multivariate_normal(mean = mean[i], 
#             cov = cov[Idx[i]:Idx[i+1],:])

#         if i ==0 :
#             X = mvn.rvs(n[i])
#             y = np.zeros(n[i])
#         else :
#             X = np.vstack((X, mvn.rvs(n[i])))
#             y = np.hstack((y, grp[i] * np.ones(n[i])))
    
    
#     return X, y


n = [200, 200, 200] # sample size for each group
# Two groups
# mean = np.array([[0.5, -0.2], [2, 2]])
# cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.]])
# Three groups
mean = np.array([[0.5, -0.2], [2, 2], [-1, 2]])
cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.], [1.0, 0.], [0., 1.]])
X, y = mvn_multiclass_data(mean, cov, n)

colors = ['b', 'r', 'g']
for i in np.arange(0, mean.shape[0]) :
    x = X[ y == i]
    plt.scatter(x[:, 0], x[:, 1], marker='o', 
        color=colors[i])

plt.grid(True)
plt.show()             
# %%
