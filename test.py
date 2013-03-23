from root_numpy._libarray2root import test
import numpy as np
a = np.array([(12345,2.,2.1),(3,4.,4.2)], dtype=[('x',np.int32),('y',np.float32),('z',np.float64)])
#test(a)
b = np.array([1,2,3.])
print type(b)
print type(np.array([]))
test(a)