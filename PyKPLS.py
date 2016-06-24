# -*- coding: utf-8 -*-
"""
Kernel Partial Least Squares Regression in Python.
"""

import numpy as np
import matplotlib.pylab as plt

def make_kernel(X,kernel):
    n=X.shape[0]
    
    if kernel=='poly2':
        K=(X.dot(X.T)+np.ones([n,n]))**2
    
    if kernel=='poly3':
        K=(X.dot(X.T)+np.ones([n,n]))**3
       
    if kernel=='rbf':
        Kd=np.diag(X.dot(X.T))
        K1=np.tile(Kd,(n,1))
        K2=K1.T
        K3=K1-2*X.dot(X.T)+K2
        
        # gamma
        gamma=np.median(K3.reshape(n**2))

        K=np.exp(-K3/gamma)
        
    return K
        
def center_kernel(K):
    n=K.shape[0]
    
    H=np.identity(n)-np.ones([n,n])/n
    K=H.dot(K).dot(H)
    
    return K
    
X=np.arange(10).reshape(5,2)
K=make_kernel(X,'rbf')
K=center_kernel(K)
plt.imshow(K)