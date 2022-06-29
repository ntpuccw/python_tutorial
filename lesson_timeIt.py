from timeit import timeit 
from datetime import datetime

setup = """
import numpy as np
a = np.arange(100000000).reshape(10000, 10000)

def contiguous_sum(x):
    for i in range(x.shape[0]):
        x[i].sum()

def non_contiguous_sum(x):
    for i in range(x.shape[-1]):
        x[:, i].sum()
"""
# contiguous_sum := sum along the row
# non_contiguous_sum := sum along the column
n=10
# starttime = timeit.default_timer()
starttime = datetime.now() # datetime.today()
print("The start time is :", starttime)
time_contiguous = timeit('contiguous_sum(a)', setup = setup, number = n) / n
time_non_contiguous = timeit('non_contiguous_sum(a)', setup = setup, number = n) / n
print("Contiguous: {:.4f}s per loop".format(time_contiguous))
print("None Contiguous: {:.4f}s per loop".format(time_non_contiguous))
print("Ratio: {:.3f}".format(time_non_contiguous / time_contiguous)) 
endtime = datetime.now() # endtime.year, endtime.month
print("The end time is :", endtime.strftime("%d/%m/%Y %H:%M:%S"))