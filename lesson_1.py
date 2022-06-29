# Basic concepts and Math operations
# start with the global namespace
# from numpy import * # import all of Numpy into the global environment
import numpy as np # numpy is a package
# import numpy.linalg as LA # linalg is a module of linear algebra 
import scipy.linalg as LA # scipy's linear algebra is better
# from numpy import *
#%%
x = [1,2,3,4,5]                 # a list 
x = np.array([1,2,5,3,4])       # a 1x5 array; full name:=ndarray()
print([x[0], x[-1]])            # first and last index
print(x[range(3)])
print(x[:3])

X = np.array([[1,2,3],[4,5,6]]) # a 2x3 matrix
print(X[0,1])                 # access a matrix
print(x.ndim)                 # check for the dimension
print(X.shape)                # (?,?)
print(X.size)                 # totla number of elements
print(X.dtype)
print(X*X)                    # elemet-wise
print(X/X)
print(X.dot(X.T))             # matrix multiplication
print(X@X.T)

A = np.arange(9).reshape(3,3) # watch the order
print(A)
print(np.eye(3))
print(np.zeros(9).reshape(3,3))

# np.matrix; not element-wise, not recommended
X = np.matrix([[1,2,3],[4,5,6]])
print(X[0,:]) # X[0,0], X[:,1]
print(np.sum(X[:,1]))
print(X[0,:].sum()) # sum the first row
# XX'
print(np.matmul(X,X.transpose())) 
print(X.dot(X.T))
print(X*X.T)        


#%% Linear Algebra stuff

A = np.array([[1, 1, 1],[2, 4, 6],[5, 7, 6]])
b = np.array([1, 2, 3])
print(np.dot(A,A)) # dot product A*A
print(np.matmul(A,A)) # matrix multiplicatoon
I = A @ A # A*A
I = A.dot(LA.inv(A))
x = LA.solve(A, b) # solve Ax=b for x
print(x)
# append a row or column to a matrix
c = np.array([7, 8, 9])
# B = np.append(A, [c], axis=0)
# B = np.row_stack((A, c))  # np.column_stack()
B = np.vstack((A, c))       # np.hstack()
print(B)
# insert a row or column in any position of a matrix
A = np.array([[1, 1, 1],[2, 4, 6],[5, 7, 6]])
b = np.array([1, 2, 3])
B_row = np.insert(A, 1, b, 0) # 1 is the position index; 0 is for row
B_col = np.insert(A, 0, b, 1) # 0 is the position index; 1 is for row
print(B_row)
print(B_col)

# python's array append approach is more efficient
pyA = [[1, 1, 1],[2, 4, 6],[5, 7, 6]] # python's array
pyc = [7, 8, 9]
pyb = [1, 2, 3, 4]
pyA.append(pyc)
print(pyA)
x = LA.lstsq(pyA, pyb) # Ax=b, x=pseudo inverse A*b
print(x[0])

#%% Logical operation
x = np.array([1, 2, 5, 3, 2])
y = np.ones(5)                # [1.,1.,1.,1.,1.]
z = np.arange(5)              # [0, 1, 2, 3, 4]
y = (x + 1) * 2               # [4, 6, 12, 8, 6]
w = (x + y) * z               # [0., 3.,12.,12.,12.]
x >= 3                        # [False, False, True, True, False]
np.where(x >= 3, x, 3) 
print(np.int32(1e8))

a = 5
print(1 < a < 10)
b = 3
a, b = b, a
print([a, b])

"tau = {0}".format(2 * np.pi) 
[2*x for x in [0, 1, 2]] # a list 

#%%
x = 2
y = x + 3
x = 3
z = x + y
print(z)

# %%
import numpy as np
a = np.arange(1000).reshape(100, 10)
a.sum() # sum all elements
print(a.sum(0)) # sum along the column
a.sum(1).shape # sum along the row 

# %%
