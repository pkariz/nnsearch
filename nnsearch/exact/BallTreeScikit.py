from ..baseindex import Index
from sklearn.neighbors import BallTree
import numpy as np
from ..datasets import Dataset


class BallTreeScikit(Index):
    """BallTree from sklearn.neighbors."""

    def __init__(self):
        self.algorithm = "BallTree-scikit"
        self.valid_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64,
                            np.float16, np.float32, np.float64]

    def build(self, data, leaf_size=20, distance="euclidean", **kwargs):
        """Builds ball tree with specified parameters.
        :param data: Dataset instance representing data
        :param leaf_size: maximum size of a leaf
        :param distance: defines metric to be used, can be "euclidean" and other values of 'metric' parameter in
        scikit's ball-tree.
        """
        if not isinstance(data, Dataset):
            raise ValueError("Data parameter must be an instance of Dataset!")
        if data.data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        self.index = BallTree(data.data, leaf_size=leaf_size, metric=distance, **kwargs)
        return self.index

    def query(self, queries, k=1):
        """Returns k nearest neighbors of each query point."""

        dists, indices = self.index.query(queries, k=k, return_distance=True)
        #return indices, dists

        if isinstance(queries, np.ndarray) and len(queries.shape) == 1 or \
                isinstance(queries, list) and not isinstance(queries[0], list):
            #return 1d arrays
            return np.array([self.index.get_arrays()[0][i] for i in indices[0]]), dists[0]
        else:
            return np.array([[self.index.get_arrays()[0][i] for i in query] for query in indices]), dists

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError
