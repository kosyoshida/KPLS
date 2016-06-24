# -*- coding: utf-8 -*-
"""
Kernel Partial Least Squares Regression in Python.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt

def single_comp(K,Y):
    n=K.shape[0]
    
    # initialize
    u=np.random.randn(n)
    t=np.random.randn(n)
    
    # iterations
    for iter in range(1000):
        t_new=K.dot(u)
        t_new/=norm(t_new,ord=2)
        c=Y.T.dot(t_new)
        u_new=Y.dot(c)
        u_new/=norm(u_new,ord=2)
        
        if norm(t_new-t,ord=2)<0.01 and norm(u_new-u,ord=2)<0.01:
            break
        else:
            u=u_new
            t=t_new  
    
    # residuals
    Hres=np.identity(n)-t_new[:,np.newaxis].dot(t_new[:,np.newaxis].T)
    Kres=Hres.dot(K).dot(Hres)
    Yres=Hres.dot(Y)
    
    return Kres,Yres,u_new,t_new
    
def make_kernel(X,kernel):
    n=X.shape[0]
    
    if kernel=='poly2':
        K=(X.dot(X.T)+np.ones([n,n]))**2
        K=center_kernel(K)
        
    if kernel=='poly3':
        K=(X.dot(X.T)+np.ones([n,n]))**3
        K=center_kernel(K)
        
    if kernel=='rbf':
        Kd=np.diag(X.dot(X.T))
        K1=np.tile(Kd,(n,1))
        K2=K1.T
        K3=K1-2*X.dot(X.T)+K2
        
        # gamma
        gamma=np.median(K3.reshape(n**2))
        
        K=np.exp(-K3/gamma)
        K=center_kernel(K)
        
    return K
        
def center_kernel(K):
    n=K.shape[0]
    
    H=np.identity(n)-np.ones([n,n])/n
    K=H.dot(K).dot(H)
    
    return K

np.random.seed(1)
    
X=np.arange(6).reshape(3,2)
K=make_kernel(X,'rbf')
Y=np.array([-1,1,1])
plt.imshow(K)
plt.show()

Kres,Yres,u,v=single_comp(K,Y)
print Kres
print Yres
print u
print v