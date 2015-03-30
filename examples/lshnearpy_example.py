from nnsearch.datasets import load_dataset
from nnsearch.approx import LSHNearPy

dataset = load_dataset("amazon")
lsh = LSHNearPy()
lsh.build(data=dataset)
query = dataset.data[0]
nearest, dists = lsh.query(query, k=1)
print "nearest:", nearest
print "distances:", dists