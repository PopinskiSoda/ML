import numpy as np
import scipy as sp
import scipy.optimize

def test_func(x):
    return (x[0])**2+(x[1])**2+285

def test_grad(x):
    return [2*x[0],2*x[1]]

myvar = sp.optimize.line_search(test_func,test_grad,np.array([1.8,1.8]),np.array([-1.,-1.]))
print(myvar)