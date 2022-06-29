'''
This program use neural network modules from neurolab
which is similar to the counterpart in MATLAB.
'''
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
from Lib_GenData import randsphere

# ------ A simple example from menu ------------
# Create train samples
# input = np.random.uniform(-0.5, 0.5, (10, 2))
# target = (input[:, 0] + input[:, 1]).reshape(10, 1)
# # Create network with 2 inputs, 5 neurons in input layer and 1 in output layer
# net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
# # Train process
# err = net.train(input, target, epochs=500, show=100, goal=0.01) # show default value
# # simulate net
# target_hat = net.sim(input)
# ---------------------------------------------
# generate training data of Robot arms
l1, l2 = 20, 10
# data distributed in a grid manner
# t = np.linspace(0, np.pi/2, 10)
# l = np.arange(l1 - l2, l1 + l2, 2)
# X = l.reshape(-1,1) @ np.cos(t.reshape(1,-1))
# Y = l.reshape(-1,1) @ np.sin(t.reshape(1,-1))
# x, y = X.reshape(-1,1), Y.reshape(-1,1)
# data distributed randomly in a circle
center = np.array([0,0])
radius_in, radius_out =10, 30
p = randsphere(center,radius_out,1000)
p = p[(p[:,0] > 0) & (p[:,1] > 0), :]
d = np.sum(p**2, axis=1)
p = p[d >= radius_in**2, :]
x, y = p[:,0], p[:,1]

theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2))
theta1 = np.arctan(y/x) - np.arctan(l2 * np.sin(theta2)/(l1 + l2 * np.cos(theta2)))

fig = plt.figure(figsize=(6,6))
plt.scatter(x, y, marker='+', alpha=0.5)

X = np.c_[x, y] # inputs: N x 2
Y = np.c_[theta1, theta2] # output: N x 2
# create network
hidden_output_layers = [20, 2] # hidden layers + output layer
transf = [nl.trans.TanSig(), nl.trans.PureLin()] # activation functions for each layer and output layer
net = nl.net.newff([[x.min(), x.max()], [y.min(), y.max()]], 
    size = hidden_output_layers, transf = transf)
#change traning func, the default training function for rrgression is train_bfgs
net.trainf = nl.train.train_bfgs # the default Using scipy.optimize.fmin_bfgs
# net.trainf = nl.train.train_cg # Newton-CG method Using scipy.optimize.fmin_ncg
# net.trainf = nl.train.train_gd
# net.trainf = nl.train.train_gdx
# net.errorf = nl.error.MSE() # default is SSE()
print(net.trainf) # show the training function 
err = net.train(X, Y, epochs = 5000, show = 100, goal = 0.01) # show := print period, the return is an error function 
Y_hat = net.sim(X)

theta1_hat, theta2_hat = Y_hat[:,0], Y_hat[:,1]
x_hat = l1 * np.cos(theta1_hat) + l2 * np.cos(theta1_hat+theta2_hat)
y_hat = l1 * np.sin(theta1_hat) + l2 * np.sin(theta1_hat+theta2_hat)
plt.scatter(x_hat, y_hat, marker='o', alpha=0.5)

mse = nl.error.MSE()
sse = nl.error.SSE()
print("Mean Square Error:{:.6f}".format(mse(Y, Y_hat)))
print("Sum Square Error:{:.6f}".format(sse(Y, Y_hat)))

plt.show()
plt.plot(err) # plot training error function: SSE
plt.show()
np.sum((Y-Y_hat)**2)/2