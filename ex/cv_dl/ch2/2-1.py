import numpy as np
 
a=np.array([4,5,0,1,2,3,6,7,8,9,10,11])
print(a)
print(type(a))
print(a.shape)
a.sort()
print(a)

b=np.array([-4.3,-2.3,12.9,8.99,10.1,-1.2])
b.sort()
print(b)

c=np.array(['one','two','three','four','five','six','seven'])
c.sort()
print(c)