# for Loop; animation plots; 
import numpy as np
import matplotlib.pyplot as plt

f=np.arange(10)
x=np.linspace(-3*np.pi,3*np.pi,200)
y=np.zeros(len(x))
plt.axis([-3*np.pi,3*np.pi, -3, 3])
plt.grid(True)
for i in f:  
    k = 2*i+1
    y = y + np.sin(k*x) /k
    plt.plot(x,y*4/np.pi,color='b')
    plt.pause(0.5)

# plt.savefig('zzz.png') # must be placed before show()
plt.show()

#%% plot function on chapter 2
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = (1-np.exp(-2*x))/(1+np.exp(-2*x))
plt.plot(x,y)
plt.show()
# %% plot function on chapter 2
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
# y = pow((4-x**3)/(1+x**2),1/3) # not correct
y = np.cbrt((4-x**3)/(1+x**2))
# y = np.cbrt(x**2)
plt.plot(x,y)
plt.grid(True)
plt.show()
# %% plot function on chapter 2
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.7, 6, 1000)
# y = np.cbrt((4-x**3)/(1+x**2))
y = np.log(x)/x**3
plt.plot(x,y)
plt.grid(True)
plt.show()
# %% Draw implicit function
# plot function on chapter 2
from sympy import plot_implicit, symbols, Eq, solve
x, y = symbols('x y')
# k=2.7
# a=3
# eq = Eq((x**2 + y**2)**2-2*a**2*(x**2-y**2), k**4-a**4)
eq = Eq(x**2+y**2,1)
p1 = plot_implicit(eq, [x,-2,2],[y,-2,2])
plt.show()
# %% Draw x^2+y^2=1 and adjust X axis and Y axis
import numpy as np 
import matplotlib.pyplot as plt 
 
angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
radius = 1
 
x = radius * np.cos( angle ) 
y = radius * np.sin( angle ) 
 
figure, axes = plt.subplots( 1 ) 
 
axes.plot( x, y ) 
axes.set_aspect( 1 ) # set x-, y-axis equal
plt.title( 'Parametric Equation Circle' ) 
# axes.axes.set_position([2,2,5,5])
axes.spines['left'].set_position(('data',0))
axes.spines['bottom'].set_position(('data',0))
axes.spines['top'].set_visible(False) # take off top line
axes.spines['right'].set_visible(False)
plt.show() 

# %%
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib


x = [1,2,2,1,1]
y = [1,1,2,2,1]
plt.plot(x,y)
plt.axis([0, 3, 0, 3])
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
fig, ax = plt.subplots(figsize=(4,3))
rec = matplotlib.patches.Rectangle((1, 1),2, 2, color ='green')
ax.add_patch(rec)
plt.xlim([0,4])
plt.ylim([0,4])
plt.show()
# %% fill patch in between two lines
import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(-0.5,1.6,100)
y1 = 2*x - x**2
y2 = x**2
plt.plot(x, y1, x, y2)
plt.fill_between(x,y1,y2, where=y1>y2, color='pink')
plt.show()
# %%
