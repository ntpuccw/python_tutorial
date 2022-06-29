# The use of sklearn.svm
import numpy as np
import seaborn as sns
from sklearn import svm
from matplotlib import cm
import matplotlib.pyplot as plt
from Lib_GenData import mvn_multiclass_data

# generate group data
# n = [200, 200, 200] # sample size for each group
# mean = np.array([[0.5, -0.2], [2, 2], [-1, 2]])
# cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.], [1.0, 0.], [0., 1.]])
n = [200, 200] # sample size for each group
mean = np.array([[0.5, -0.2], [3, 3]])
cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.]])
X, y = mvn_multiclass_data(mean, cov, n)

svm_clf = svm.SVC(C = 1, kernel='linear', verbose=True)
svm_clf.fit(X, y)
# Check with some output parameters
svm_clf.class_weight_
svm_clf.support_vectors_
svm_clf.intercept_
svm_clf.coef_ # for linear kernel
svm_clf.classes_
svm_clf.dual_coef_
svm_clf.get_params()

# get the separating hyperplane: b0 + b1*x1 + b2*x2 + ... =0
b0, b1, b2 = svm_clf.intercept_[0], svm_clf.coef_[0][0], svm_clf.coef_[0][1] 
xx = np.linspace(-5, 6)
yy = -b1 / b2 * xx - b0 / b2

plt.figure()
plt.plot(xx, yy)
# plot the parallels to the separating hyperplane that pass through the
# support vectors (margin away from hyperplane in direction
# perpendicular to hyperplane). 
margin = 1 / np.sqrt(np.sum(svm_clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + (-b1/b2) ** 2) * margin
yy_up = yy + np.sqrt(1 + (-b1/b2) ** 2) * margin
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(svm_clf.support_vectors_[:, 0], 
        svm_clf.support_vectors_[:, 1], s=80,
        facecolors='none', zorder=10, edgecolors='k',
        cmap=cm.get_cmap('RdBu'), alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, 
        cmap=cm.get_cmap('RdBu'), edgecolors='k', alpha=0.5)

print("accuracy by score in train: \
     {:.2f}%".format(100*svm_clf.score(X, y)))

# target_names = np.array(['A', 'B', 'C'])
# target_names = np.array(['A', 'B'])
# # cmap_bold = ['darkorange', 'c', 'darkblue']
# cmap_bold = ['darkorange', 'c']

# sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=target_names[y], palette=cmap_bold,
#                  alpha=1.0, edgecolor="black")
x_min, x_max = plt.xlim() # get the x limits of the current axes
y_min, y_max = plt.ylim() 
xx = np.linspace(x_min, x_max,20)
yy = np.linspace(y_min, y_max,20)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_clf.decision_function(xy).reshape(XX.shape)

# Put the result into a contour plot
plt.contourf(XX, YY, Z, cmap=cm.get_cmap('RdBu'),
                 alpha=0.5, linestyles=['-'])
plt.title('The linear hyperplane of SVM')
plt.colorbar()
plt.show()

# %%
# Also check with the example in sklearn
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py
# use the decision_function values(-0.5 0 0.5) to determine the boundary
