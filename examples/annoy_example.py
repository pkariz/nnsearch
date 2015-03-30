from nnsearch.datasets import Dataset
from nnsearch.approx import Annoy
import numpy as np

a = Annoy()
rows = []
rows.append([1, 0, 0])
rows.append([0, 10, 20])
rows.append([2, 2, 2])
ds = Dataset(data=np.array(rows))
metric = "angular"
a.build(data=ds, dimensions=3, metric=metric)
print "a.metric:", a.metric
print a.query(np.array([1.0, 0.5, 0.5]), k=2)
a.save('test.tree')
print "nn:", a.index.get_item_vector(0)
print "dist:", a.index.get_distance(0, 1)
print a.get_dist(a.index.get_item_vector(0), [1.0, 0.5, 0.5])
b = Annoy()
b.load('test.tree', a.d, metric)

print b.query(np.array([10.0, 5.0, 15.0]), k=2)
print b.query(np.array([np.array([10.0, 5.0, 15.0]), np.array([1.0, 0.5, 0.5])]), k=2)

