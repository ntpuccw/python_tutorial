# Data type in Python
#%% 1. Numbers

int_number = 1
float_number = 1.0
complex_number = 1 + 2j
str2int = int('213')
round_float = round(1234.5678, 2)

print(complex_number)
# %% 2. string
# the following strings definitions are the same for single line string
s1 = 'a string'
s2 = "also a string"
s3 = '''A multiline string''' #  triple quotations are used to assign strings with multiple lines .
print(type(s2))
print(s3[0])
print(s2[:4])
print(s3)
s4 = """
Triple double quotes
is used
in mulitiple lines.
"""
print(s4)
# concatenate strings
s5 = 'This is ' + s1
print(s5)
# %% 3. list := cellarray in MATLAB
# array := matrix in MATLAB
# use array when arithmetic computation is necessary
# indexing and slicing

word_list = ['the', 'quick', 'brown', 'fox']
number_list = [0, 1, 1, 2, 3, 5, 8]
print(word_list[3]) # indexing 
print(number_list[0:3]) # slicing from index n to the mth position of the list
print(word_list[-1]) # the last one
print(word_list[-3:]) # the last three

# the slicings use the zero-based indexing, 
# that includes the lower bound and omit the upper bound, e.g.
a = [1, 2, 3, 4, 5, 6, 7] # = np.arange(1, 8)
low, high = 2, 4
print( a[0:low] + a[low:high] + a[high:7]) # this explain why the indeing starts from 0

import numpy as np
print(np.size(number_list))
print(np.shape(number_list))
a = [[1, 2, 3],[4, 5, 6]]
print(np.size(a)) # number of elements
print(np.shape(a)) # array dimension
print(a[2])

a = np.array([1, 2, 3]) # a vector
A = np.array([[1, 2, 3]]) # an 1 x 3 array that can be transposed
B = a[:, np.newaxis] # a'
C = np.array([1, 2, 3], ndmin = 2)
a.shape
A.shape
B.shape
C.T.shape

a = np.array([[1, 2, 3],[4, 5, 6]])
print(a.shape) # shape() is a method of a numpy array
print(a[1])
print(a[1,:])
print(a[:,1])
print(a[:,[0,2]])

a[a < 3] = 0
print(a)
b = a[:, 2]
c = a.flatten() # turn into vector
d = a.flatten('F') # turn into vector column-wise
a = np.r_[2 : 12] # = np.arange(2:12), can not be done like [2:12]
print(a) 

Z = np.zeros((3,4))
One =  np.ones((3,4))
I = np.eye(3)
D = np.diag(a) # a is a vector --> D a diagonal matrix
e = np.diag(I) # I is a square matrix --> e a vector of the diagonal of I

a = [1, 2, 3, 4]
np.tile(a,(5,1)) # = repmat in MATLAB
b = np.reshape(a, (-1, 1)) # convert to 4 x 1

# vector concatenate
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate((a, b), axis = 0) # flatten a, b
np.append(a, b)
np.hstack((a,b)) 
np.r_[a, b]

print((a, b)) # this is matrix concatenate

#matrix concatenate
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])
# Vertically, concatenate rows 
np.concatenate((a, b), axis = 0) # flatten a, b
np.r_[a, b]
np.vstack((a,b))
# concatenate column
np.concatenate((a.T, b.T), axis =1)
np.hstack((a.T, b.T))
np.c_[a.T, b.T]

C = np.concatenate((a, b), axis = 0)
C.max()
C.max(0) # column max
C.max(1) # max

# Arrays and lists are both used in Python to 
# store data, but they don't serve exactly the 
# same purposes. They both can be used to store 
# any data type (real numbers, strings, etc), 
# and they both can be indexed and iterated 
# through, but the similarities between the 
# two don't go much further. The main difference 
# between a list and an array is the functions 
# that you can perform to them. For example, 
# you can divide an array by 3, and each number 
# in the array will be divided by 3 and the 
# result will be printed if you request it. 
# If you try to divide a list by 3, Python will 
# tell you that it can't be done, and an error 
# will be thrown.
# %% 4. tuple

point_tuple = (0, 0) # ordered sequence
also_a_point_tuple = 0, 0
a_3d_point_tuple = (0, 1, 2) # (2,1,0) is different, however the list  [0,1,2] := [1,2,0]

# 5. Dictionaries similar to structure in MATLAB
meals = {'breakfast': 'sardines', 'lunch': 'salad', 'dinner': 'cake', 'cost':40}
print(meals['breakfast'])
print(meals['cost'])

# 6. sets
lights = {'red', 'yellow', 'green'}
choices = ['yes', 'no', 'yes', 'yes', 'no']
unique_choices = set(choices)
print(unique_choices)
print(len(unique_choices))
print(type(unique_choices))
color="red"
print(color in lights)
print( lights | unique_choices) # or
set1 = {"abc", 34, True, 40, "male"}
# %%
import sys
print(sys.path)
# %%
