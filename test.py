import numpy as np

a=[[1,2,3],[4,5,6]]
b=[[1,2,3],[4,5,6]]

a=np.asanyarray(a)
b=np.asanyarray(b)
c=np.sum(a*b, axis=1)
print (c)