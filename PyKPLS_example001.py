# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:31:33 2016

@author: yoshidakosuke
"""

import numpy as np
import PyKPLS as pls

np.random.seed(1)
    
X=np.arange(6).reshape(3,2)
K=pls.make_kernel(X,'rbf')
Y=np.array([-1,1,1])
U,T,C=pls.KPLS(K,Y,k=2)
print U
print T
print C