'''
Implement articifical neural network for classification

-- Employ the dataset of mnist digits
-- Classification by sklearn.neural_network.MLPClassifier
-- Separate training  and testing data by sklearn.model_selection
-- Plot confusion matrix by sklearn.metrics
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
# from sklearn.linear_model import Perceptron
from scipy.io import loadmat, savemat

# each digit image is 28x28
# X, y = fetch_openml('mnist_784', version =1, return_X_y = True)
# each digit image is 8x8
# digits = datasets.load_digits()
data_dir = '../Data/'
D = loadmat(data_dir + 'Digits_train.mat')
# D.keys()
X = D['X'] # images
y = D['y'] # labels: single output in 0~9
plt.figure()
# Draw a montage of letters
n, m = 20, 30  # A m x n montage (total mn images)
sz = np.sqrt(X.shape[1]).astype('int') # image size sz x sz
M = np.zeros((m*sz, n*sz))
A = X[:m*n,:]
# Arrange images to form a montage  
for i in range(m) :
    for j in range(n) :
        M[i*sz: (i+1)*sz, j*sz:(j+1)*sz] = \
            A[i*n+j,:].reshape(sz, sz)

plt.imshow(M.T, cmap=plt.cm.gray_r, interpolation='nearest')
plt.xticks([])
plt.yticks([])
plt.title('The Montage of digit letters')
plt.show()

hidden_layers = (30,) # one hidden layer
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y.ravel(), test_size = 0.25)
# solver = 'sgd' # not efficient, need more tuning
# solver = 'lbfgs' # not suitable here
solver = 'adam' # default solver
mlpClassifier = MLPClassifier(max_iter = 10000, solver = solver,
    hidden_layer_sizes = hidden_layers, verbose = True,
    activation = 'logistic', tol = 1e-6, random_state = 0) # default activation = 'relu'
mlpClassifier.fit(X_train, y_train)
predicted = mlpClassifier.predict(X_test)
print("accuracy for tested data: {:.2f}%".format(100*np.mean(predicted == y_test)))
print("accuracy by score for tested data: {:.2f}%".format(100*mlpClassifier.score(X_test, y_test)))
# check out the model structure
mlpClassifier.loss_  # The current loss computed with the loss function
mlpClassifier.best_loss_
mlpClassifier.n_layers_  # input layer is counted
mlpClassifier.n_outputs_ # Number of outputs.
mlpClassifier.out_activation_ # softmax is employed here
mlpClassifier.n_iter_ # The number of iterations the solver has run.
mlpClassifier.t_ # The number of training samples seen by the solver during fitting
mlpClassifier.classes_ # Class labels for each output.
mlpClassifier.get_params(deep=True) # get all parameters for this estimator

plt.plot(mlpClassifier.loss_curve_)
plt.grid(True)
plt.title('Training Loss Curve')
plt.xlabel('Iter.')
plt.ylabel('Fitting Loss')
plt.show()


# Confusion matrix
plot_confusion_matrix(mlpClassifier, X_test, y_test,
    cmap=plt.cm.Blues, normalize='true')  # normalize='false' to show numbers
plt.title('Normalized Confusion matrix for tested data')
plt.show() 