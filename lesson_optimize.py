# Unconstraint Optimization similar to fminsearch in MATLAB
# Constraint Optimization similar to fmincon in MATLAB
# %% Unconstraint Optimization by fmin
import scipy.optimize as opt

# use lambda to express in-line function
banana = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
OptVal = opt.fmin(func=banana, 
    x0=[-1.2,1], 
    maxiter=1e3, 
    maxfun=1e3, 
    disp=0, 
    full_output=True)

xopt = OptVal[0] # x1 = OptVal[0][0], x2 = OptVal[0][1]
fopt = OptVal[1]
print([xopt,fopt])

#%%
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

opts = dict(disp=1, xtol=1e-6, ftol=1e-6, maxfun=1e4, maxiter=1e4, full_output=True)
f = lambda x: (x[0] - 2)**4 + (x[0] - 2)**2*x[1]**2 + (x[1]+1)**2
OptVal = opt.fmin(func=f, x0=[0, 0], **opts)
xopt = OptVal[0]
fopt = OptVal[1]
print([xopt,fopt])

# Make a 3D graph
x = np.linspace(1, 3, 100)
y = np.linspace(-2, 0, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_wireframe(X, Y, Z, color ='blue',alpha=0.6)
ax.view_init(15, -59)  #(elev, azim=)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
# %% Constraint Optimization by minimize
# Exercise 9.4 on page 201
import scipy.optimize as opt
import numpy as np

opts = dict(disp = True, maxiter=1e4)
# set constraints
# (1) x1>0, x2>0; or use bounds = [(0, None), (0, None)]
# cons = [{'type': 'ineq', 'fun': lambda x:  x[0]},
#         {'type': 'ineq', 'fun': lambda x:  x[1]}]

# (2) x1<=1, x2<=2; or use bounds = [(0, 1), (0, 2)]
# bnds = [(None, 1), (None, 2)]
# cons = [];
# cons = [{'type': 'ineq', 'fun': lambda x:  1 - x[0]},
#         {'type': 'ineq', 'fun': lambda x:  2 - x[1]}]
# (3) 0<=x1<=1, 0<=x2<=2
# bnds = [(0, 1), (0, 2)]
# cons =[]
# (4) 0<=x1<=inf, -inf<=x2<=2
# bnds = [(0, np.inf), (-np.inf, 2)]
# cons =[]
# (5) x1+x2<=0.9
# bnds = []
# cons = {'type': 'ineq', 'fun': lambda x:  -x[0] - x[1] + 0.9}
# (6) 1.5<=x1+x2<=2
# bnds = []
# method 1
# cons = [{'type': 'ineq', 'fun': lambda x:  x[0] + x[1] - 1.5},
#         {'type': 'ineq', 'fun': lambda x:  -x[0] - x[1] + 2}]
# method 2
# A = [[1, 1],[-1, -1]]
# b = [1.5, -2]
# cons = {'type': 'ineq', 'fun': lambda x: A @ x - b}
# method 3: b <= Ax <= inf
# lb = b
# ub = np.inf*np.ones(2)
# cons = opt.LinearConstraint(A, lb, ub)
# (7) sqrt(x1**2 + x2**2)<=1
# bnds = []
# con = lambda x: np.sqrt(x[0]**2 + x[1]**2) 
# lb = 0
# ub = 1
# cons = opt.NonlinearConstraint(con, lb, ub)

# (8) sqrt(x1**2 + x2**2)<=1, 0 <= x1*x2
cons = [{'type': 'ineq', 'fun': lambda x:  
            1 - np.sqrt(x[0]**2 + x[1]**2)},
        {'type': 'ineq', 'fun': lambda x:  
            x[0] * x[1]}]

bnds = []
f = lambda x: (x[0] - 2)**4 + (x[0] - 2)**2*x[1]**2 + (x[1]+1)**2
res = opt.minimize(f, x0=[0, 0], 
    bounds = bnds,
    constraints = cons,
    options = opts,
    tol = 1e-8)
print(res)
  
# %% create bounds from A
import numpy as np
A = np.array([[1, -2], [-1, -2], [-1, 2]])
bnds = [(0, None) for i in range(A.shape[1])]
print(bnds)
# %%
