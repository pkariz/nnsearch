from nnsearch.datasets import Dataset
from nnsearch.approx import HKmeans
import numpy as np

dataset = np.array([np.array([1,0,0,1]), np.array([1,1,1,1]), np.array([0,0,10,10])], dtype=np.uint8)
testset = np.array([1,1,0,1], dtype=np.uint8)
ds = Dataset(data=dataset)
print "ds.data:", ds.data
tree = HKmeans()
tree.build(ds)
result, dists = tree.query(testset, k=1)

print "result1:", result
print "dists1:", dists

tree.save("kmeans.data")

print "tree saved!"

tree2 = HKMeans()
tree2.load("kmeans.data",ds)

print "tree loaded"
result, dists = tree2.query(testset,k=2)
print "result:", result
print "dists:", dists