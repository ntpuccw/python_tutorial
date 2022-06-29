import numpy as np
from scipy.special import gammainc
from scipy.stats import multivariate_normal


def mvn_multiclass_data(mean, cov, n):
    """
    generate multiclass data with multivariate Normal dist.
    Ex. for a 3 class data
    n = [200, 200, 200] # sample size for each group
    mean = np.array([[0.5, -0.2], [2, 2], [-1, 2]])
    cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.], [1.0, 0.], [0., 1.]])
    """
    # X = np.array([])
    # y = np.array([])
    grp_size = mean.shape[0]
    grp = np.arange(0, grp_size)
    Idx = np.arange(0, grp_size * 2 + 1, 2)
    for i in grp:
        mvn = multivariate_normal(mean=mean[i], cov=cov[Idx[i] : Idx[i + 1], :])

        if i == 0:
            X = mvn.rvs(n[i])
            y = np.zeros(n[i])
        else:
            X = np.vstack((X, mvn.rvs(n[i])))
            y = np.hstack((y, grp[i] * np.ones(n[i])))

    y = y.astype(int)  # convert to integer for class
    return X, y

n = [300, 200, 100] # sample size for each group
mean = np.array([[0.5, -0.2], [2, 2], [-1, 2]])
cov = np.array([[2.0, 0.3], [0.3, 0.5], [1.0, 0.], [0., 1.], [1.0, 0.], [0., 1.]])
X, y = mvn_multiclass_data(mean, cov, n)


def randsphere(center, radius, n_per_sphere):
    """generate random numbers in a n-dimensional sphere
    i.e. in 2D, it is in a circle; in 3D, it is in a ball
    """
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sum(x ** 2, axis=1)
    fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_per_sphere, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    return p
