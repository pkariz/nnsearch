from ..baseindex import Index
from ..datasets import Dataset
from ..algos.BoundaryForest import BoundaryForest
import numpy as np


class BF(Index):
    """Implementation of PM-tree."""

    def __init__(self):
        self.algorithm = "BoundaryForest"
        self.valid_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64,
                            np.float16, np.float32, np.float64]
        self.index = None
        self.size = None

    def build(self, data, labels=None, trees=4, max_node_size=50, task="knn", distance="euclidean", dc="euclidean",
              eps=None, parallel=True, n=5):
        """
        Builds boundary trees with specified parameters.
        :param data: training data (instance of Dataset), size of data must be >= number of trees
        :param labels: labels of data
        :param trees: number of boundary trees
        :param max_node_size: node capacity
        :param task: can be "knn", "regression" or "classification"
        :param d: distance used between two positions, in case of "knn" task positions are also labels
        :param dc: distance used between two labels, in case of "knn" task this is ignored since 'd' is used
        :param eps: defines an error window for regression problems. If a query result is > eps away from true value in
        a boundary tree then the estimate was wrong and it will create a new node with this 'missed' example in this
        tree.
        :param n: number of random dimensions to use for distance computations, default is 5
        """
        d = distance
        if not isinstance(data, Dataset) or data.data is None:
            raise ValueError("Invalid data. Should be an instance of Dataset and contain some data!")
        if type(data) is np.ndarray and data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        self.index = BoundaryForest(data.data, labels=labels, trees=trees, max_node_size=max_node_size, task=task,
                                    d=d, dc=dc, eps=eps, parallel=parallel, n=n)
        self.index.create_trees()

    def insert(self, x, x_label=None):
        """Inserts x in structure."""
        if x_label is None:
            if self.index.task != 0:
                raise ValueError("Label is missing!")
            else:
                x_label = x
        if not isinstance(x, np.ndarray) or len(x.shape) != 1:
            raise ValueError("Invalid parameter, must be 1d numpy array!")
        self.index.train(x, x_label)

    def query(self, queries, k=1):
        if isinstance(queries, np.ndarray) and len(queries.shape) == 1 or \
            isinstance(queries, np.ndarray) and len(queries.shape) == 2:
            ns, ds = self.index.query(queries, k, parallel=False)
            return ns, ds
        else:
            raise ValueError("Invalid shape or type of parameter 'queries'!")

    def save(self, filename):
        """Saves BF to file"""
        self.index.save(filename=filename)

    def load(self, filename, data, labels=None, distance=None, dc=None):
        """Loads BF-tree from file, requires data and optionally labels, d and dc"""
        if isinstance(data, np.ndarray):
            np_data = data
        elif isinstance(data, Dataset):
            np_data = data.data
        else:
            raise ValueError("Invalid data parameter!")
        self.index = BoundaryForest(data=np.array([[0.0, 0.1]]), trees=1).load(filename, np_data, labels=labels,
                                                                                d=distance, dc=dc)
        self.size = self.index.size
