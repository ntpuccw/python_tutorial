# Make a 3D graph including mesh (wireframe), surface, contour graphs
#%%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X * np.exp(-X**2 - Y**2)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_wireframe(X, Y, Z,  
        rstride=2, cstride=2)
# ax.plot_wireframe(X, Y, Z,  rstride=0, cstride=10)
# ax.plot_wireframe(X, Y, Z,  rstride=0, cstride=10)
ax.view_init(13, -60)  #(elev=-165, azim=60)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(9, 6), 
    subplot_kw={'projection': '3d'})

ax1.plot_wireframe(X, Y, Z,  rstride=10, cstride=0) # y fixed
ax2.plot_wireframe(X, Y, Z,  rstride=0, cstride=10) # x fixed
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.tight_layout()      # tight up subplots
plt.show()
# %%
# repeat colors (r,g,b) from lowest level to highest level
levels = [-1, -0.4, -0.2, 0, 0.2, 0.4, 1]
C1 = plt.contourf(X, Y, Z, levels, 
        colors =('r', 'g', 'b'), 
                origin = 'lower', 
                        extend ='both')

C1.cmap.set_under('yellow') # data below the lowest contour
C1.cmap.set_over('yellow') # data over the highest level
C2 = plt.contour(C1, levels,
                  colors =('k', ),
                  linewidths =(3, ),
                  origin = 'lower')

plt.clabel(C2, fmt ='% 2.1f', 
             colors ='w',
             fontsize = 14)
cbar = plt.colorbar(C1)

plt.show()

# %% Use registered colormap 
levels = [-1, -0.4, -0.2, 0, 0.2, 0.4, 1]
cmap = plt.cm.get_cmap('winter').copy() # make a copy to not alter the registered one
cmap.set_under("green")
cmap.set_over("red")
C1 = plt.contourf(X, Y, Z, levels, 
        cmap = cmap,
                origin = 'lower', 
                        extend ='both')

C2 = plt.contour(C1, levels,
                  colors =('k', ),
                  linewidths =(3, ),
                  origin = 'lower')

plt.clabel(C2, fmt ='% 2.1f', 
             colors ='w',
             fontsize = 14)
cbar = plt.colorbar(C1)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: (x[0] - 2)**4 + (x[0] - 2)**2*x[1]**2 + (x[1]+1)**2

x = np.linspace(1, 3, 100)
y = np.linspace(-2, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
fig = plt.figure()
ax = plt.axes(projection ='3d')
surf1 = ax.plot_surface(X, Y, Z, 
        cmap='ocean',
        rstride=8, 
        cstride=8, 
        alpha=0.8) # cmap='coolwarm', 'gist_earth','ocean'
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
# The keyword arguments rstride= and cstride= determine the row step size and the column step size.
# ax.plot_wireframe(X, Y, Z, 
#         color ='blue',
#         alpha=0.6,
#         rstride=2, cstride=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DEMO')
plt.show()

plt.contour(X, Y, Z, 100) 
plt.show()

levels = np.arange(0.1,5,0.2)
contours = plt.contour(X, Y, Z,levels=levels) # check dir(contours)
plt.clabel(contours, inline=1, fontsize=10) # inline =1 or 0 
plt.grid(True)
plt.show()
# %%
C = plt.contourf(X, Y, Z, 20, \
        cmap = plt.cm.bone)

C2 = plt.contour(C, levels=C.levels, colors ='r')
cbar = plt.colorbar(C)
plt.xlabel('X')
plt.ylabel('Y')
cbar.ax.set_ylabel('Z = f(X,Y)') # set colorbar label
cbar.add_lines(C2) # add contour line levels to the colorbar 
plt.show()
# %%
