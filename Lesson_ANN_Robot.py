import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import plot_confusion_matrix

l1, l2 = 20, 10
t = np.linspace(0, np.pi/2, 10)
l = np.arange(l1 - l2, l1 + l2, 2)
X = l.reshape(-1,1) @ np.cos(t.reshape(1,-1))
Y = l.reshape(-1,1) @ np.sin(t.reshape(1,-1))

x, y = X.reshape(-1,1), Y.reshape(-1,1)
theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2))
theta1 = np.arctan(y/x) - np.arctan(l2 * np.sin(theta2)/(l1 + l2 * np.cos(theta2)))

fig = plt.figure(figsize=(6,6))
plt.scatter(x, y, marker='+', alpha=0.5)

# check the formula
# x_hat = l1 * np.cos(theta1) + l2 * np.cos(theta1+theta2)
# y_hat = l1 * np.sin(theta1) + l2 * np.sin(theta1+theta2)
# plt.scatter(x_hat, y_hat, marker='o', alpha=0.5)
# plt.show()

# setup for ANN training
X = np.c_[x, y]
Y = np.c_[theta1, theta2]
hidden_layers = (100, )
mlp_reg = MLPRegressor(max_iter=1000, solver='lbfgs',
    hidden_layer_sizes=hidden_layers, verbose = False,
    activation = 'logistic', 
    tol=1e-6, random_state=0) # default activation = 'relu'
mlp_reg.fit(X, Y)
Y_hat = mlp_reg.predict(X)

theta1_hat, theta2_hat = Y_hat[:,0], Y_hat[:,1]
x_hat = l1 * np.cos(theta1_hat) + l2 * np.cos(theta1_hat+theta2_hat)
y_hat = l1 * np.sin(theta1_hat) + l2 * np.sin(theta1_hat+theta2_hat)

# x_hat = l1 * np.cos(Y_hat[:,0]) + l2 * np.cos(Y_hat[:,0]+Y_hat[:,1])
# y_hat = l1 * np.sin(Y_hat[:,0]) + l2 * np.sin(Y_hat[:,0]+Y_hat[:,1])
plt.scatter(x_hat, y_hat, marker='o', alpha=0.5)
plt.show()
# R_square
print("R square in training: {:.2f}%".format(mlp_reg.score(X, Y)))
mlp_reg.n_layers_
mlp_reg.n_outputs_
mlp_reg.out_activation_
mlp_reg.get_params(deep=True)
# mlp_reg.loss_curve_

# from sklearn import cross_validation 
# cross_validation.cross_val_score(mlp_reg, X, Y,scoring='mean_squared_error')

# if solver='lbfgs', loss_curve_ attribute is not available
# plt.plot(mlp_reg.loss_curve_)
# plt.grid(True)
# plt.title('Training Loss Curve')
# plt.xlabel('Iter.')
# plt.ylabel('Fitting Loss')
# plt.show()
