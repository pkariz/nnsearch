from nnsearch.datasets import Dataset
from nnsearch.exact import BallTree
import numpy as np

dataset = np.array([np.array([1, 0, 0, 1]), np.array([1, 1, 1, 1]), np.array([0, 0, 10, 10])], dtype=np.uint8)
testset = np.array([np.array([1, 1, 0, 1]), np.array([0, 0, 9, 10])], dtype=np.uint8)

ds = Dataset(data=dataset)
print "ds.data:", ds.data
tree = BallTree()
tree.build(ds)
result, dists = tree.query(testset, k=2)

print "result:", result
print "dists:", dists