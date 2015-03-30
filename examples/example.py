from nnsearch.datasets import load_dataset
from nnsearch.exact import RTree

dataset = load_dataset("german_post_codes")

#create and build r-tree
rtree = RTree()
rtree.build(data=dataset)

#plot r-tree
rtree.plot()

#query single point
query = dataset.data[0]
neighbors, distances = rtree.query(query, k=5)
print "neighbors:", neighbors
print "distances:", distances