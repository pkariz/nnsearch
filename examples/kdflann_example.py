from nnsearch.datasets import Dataset
from nnsearch.exact import KDTree
import numpy as np

dataset = np.array([np.array([10, 0, 0, 1]), np.array([15, 1, 1, 1]), np.array([30, 0, 10, 10])], dtype=np.float64)
testset = np.array([np.array([4 ,1, 0, 1]), np.array([20, 0, 9, 10])], dtype=np.float64)


ds = Dataset(data=dataset)
print "ds.data:", ds.data
tree = KDTree()
tree.build(ds, max_leaf_size=10)
result, dists = tree.query(testset, k=2)

print "result:", result
print "dists:", dists

tree.save("kdtree.data")

print "tree saved!"

tree2 = KDTree()
tree2.load("kdtree.data",ds)

print "tree loaded"
result, dists = tree2.query(testset, k=2)
print "result:", result
print "dists:", dists

