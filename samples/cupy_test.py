import numpy as np
import cupy as cp
import time
 
x=np.ones((1024,512,4,4))*1024.
y=np.ones((1024,512,4,1))*512.3254
time1=time.time()
for i in range(20):
    z=x*y
print('average time for 20 times cpu:',(time.time()-time1)/20.)
 
x=cp.ones((1024,512,4,4))*1024.
y=cp.ones((1024,512,4,1))*512.3254
time1=time.time()
for i in range(20):
    z=x*y
print('average time for 20 times gpu:',(time.time()-time1)/20.)
