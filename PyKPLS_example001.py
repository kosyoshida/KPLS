# -*- coding: utf-8 -*-
"""
example file for PyKPLS
"""

import numpy as np
import PyKPLS as pls
import matplotlib.pylab as plt
from sklearn import preprocessing

np.random.seed(1)

# data    
x=np.linspace(-10,10,num=100)
y=np.sin(np.abs(x))/np.abs(x)
y=preprocessing.scale(y)

# kernel
K=pls.make_kernel(x,'rbf')

# perform kernel pls
D,U,T,C=pls.KPLS(K,y,k=3)

yhat=K.dot(D)

plt.figure()
plt.plot(x,y,':b',label='original data')
plt.plot(x,yhat,'-r',label='prediction data')
plt.legend(loc='best')
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.show()

plt.figure()
plt.plot(x,T[:,0],':r',linewidth=2,label='1st latent component')
plt.plot(x,T[:,1],':g',linewidth=2,label='2nd latent component')
plt.plot(x,T[:,2],':k',linewidth=2,label='3rd latent component')
plt.legend(loc='best')
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.show()