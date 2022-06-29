''' 
Implement K-nearest neighbors for classification

-- KNN classification by sklearn.neighbors
-- scatter plot by seaborn
-- colormap by matplotlib.colors
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

K = 5 # k nearest neighbors
iris = datasets.load_iris() 
X = iris.data[:,:2] # use the first two colomns
y = iris.target
# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

weights = 'uniform' # 'distance'
Knn = neighbors.KNeighborsClassifier(K, weights=weights)
Knn.fit(X, y)

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1  
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1  

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                    np.arange(y_min, y_max, 0.02))

Z = Knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
                palette=cmap_bold, alpha=1.0, edgecolor="black")
plt.title("3-Class classification (k = %i, weights = '%s')"
              % (K, weights))

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()                

# %% KNN on simulated data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from Lib_GenData import mvn_multiclass_data

K = 5 # k nearest neighbors
n = [200, 200, 200] # sample size for each group
mean = np.array([[0.5, -0.2], [2, 2], [-1, 2]])
cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.], [1.0, 0.], [0., 1.]])
X, y = mvn_multiclass_data(mean, cov, n)
# y = y.astype(int) # convert to integer

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

weights = 'uniform' # 'distance'
Knn = neighbors.KNeighborsClassifier(K, weights=weights)
Knn.fit(X, y)

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1  
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1  

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                    np.arange(y_min, y_max, 0.02))

Z = Knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)
target_names = np.array(['A', 'B', 'C'])
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=target_names[y],
                palette=cmap_bold, alpha=1.0, edgecolor="black")
plt.title("3-Class classification (k = %i, weights = '%s')"
              % (K, weights))

plt.xlabel('X1')
plt.ylabel('X2')
plt.show()        
# save data to and load data from a txt file
np.savetxt('demo_data.txt', np.c_[X, y], 
     fmt="%.3f %.3f %d", header="X1 X2 y")
Data = np.loadtxt('demo_data.txt')

# %%
