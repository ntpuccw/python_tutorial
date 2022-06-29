# %%
import numpy as np
from scipy.special import gammainc
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
from Lib_GenData import randsphere

# def randsphere(center,radius,n_per_sphere):
#     r = radius
#     ndim = center.size
#     x = np.random.normal(size=(n_per_sphere, ndim))
#     ssq = np.sum(x**2,axis=1)
#     fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
#     frtiled = np.tile(fr.reshape(n_per_sphere,1),(1,ndim))
#     p = center + np.multiply(x,frtiled)
#     return p

fig1 = plt.figure(1)
ax1 = fig1.gca()
center = np.array([0,0])
radius = 1
x = randsphere(center,radius,1000)
ax1.scatter(x[:,0],x[:,1],s=0.5)
# circle = plt.Circle(center,radius,fill=False,color='0.5')
circle = Circle(center,radius,fill=False,color='0.5')
# ax1.add_artist(circle)
ax1.add_patch(circle)
ax1.set_xlim(-1.5,1.5)
ax1.set_ylim(-1.5,1.5)
ax1.set_aspect('equal')
plt.show()
#%% random numbers in the constrained area 

fig2 = plt.figure(1)
ax2 = fig2.gca()
center = np.array([0,0])
radius_in, radius_out =10, 30
x = randsphere(center,radius_out,1000)
# Idx1 = x[:,0] > 0 #and 
# Idx2 = x[:,1] > 0
# Idx = Idx1 & Idx2
# x = x[Idx, :]
x = x[(x[:,0] > 0) & (x[:,1] > 0), :]
d = np.sum(x**2, axis=1)
x = x[d >= radius_in**2, :]
ax2.scatter(x[:,0],x[:,1],s=0.5)
ax2.add_artist(plt.Circle(center, radius_out,
        fill=False, color='0.5'))
ax2.add_artist(plt.Circle(center, radius_in,
        fill=False, color='0.5'))        
# ax2.set_xlim(-1.5,1.5)
# ax2.set_ylim(-1.5,1.5)
t = np.linspace(0, np.pi/2, 10)
x1, y1 = radius_out * np.sin(t), radius_out * np.cos(t) 
s = np.linspace(np.pi/2, 0, 10)
x2, y2 = radius_in * np.sin(s), radius_in * np.cos(s) 
x, y = np.append(x1, x2), np.append(y1, y2)
X = np.c_[x, y]
# get a patch object
fan = Polygon(X, facecolor='g', 
        edgecolor='r', alpha=0.3)
ax2.add_patch(fan)
ax2.set_aspect('equal')
plt.show()


# %%
import numpy as np
from Lib_GenData import randsphere
from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=[6, 6])
ax = fig.add_subplot(111, projection='3d')

# ax.set_aspect('equal')
# generate random numbers in a sphere
center = np.array([0, 0, 0])
radius = 1
p = randsphere(center, radius, 1000)
ax.scatter3D(p[:,0], p[:,1], p[:,2], s=1, color = 'b')

# Draw a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z,  rstride=4, cstride=2, 
        color='r', linewidth=0, alpha=0.5)
ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z))) # ptp(x) =x.max()-x.min()
plt.show()

# %%
