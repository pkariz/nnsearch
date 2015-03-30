from nnsearch.datasets import Dataset
from nnsearch.approx import LSHFlann
import numpy as np

dataset = np.array([np.array([1,0,0,1]), np.array([1,1,1,1]), np.array([0,0,0,1])], dtype=np.uint8)
testset = np.array([np.array([1,0,0,2]), np.array([5,0,4,1])], dtype=np.uint8)

ds = Dataset(data=dataset)
print "ds.data:", ds.data
lsh = LSHFlann()
lsh.build(ds)
result, dists = lsh.query(testset,k=2)

print "result:", result
print "dists:", dists