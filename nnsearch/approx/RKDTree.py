from ..flannindex import FlannIndex
from ..datasets import Dataset
import numpy as np


class RKDTree(FlannIndex):
    """Randomized KDTree algorithm from FLANN."""

    def __init__(self):
        super(RKDTree, self).__init__()
        self.algorithm = "RKD-Tree-flann"
        self.valid_types = [np.uint8,
                            np.int32,
                            np.float32, np.float64]

    def build(self, data, trees=4, precision=None, **kwargs):
        """Builds flann's kdtree index.
        :param data: Dataset instance representing data
        :param trees: number of randomized kdtrees, are searched in parallel
        :param precision: the desired precision of searches
        """
        if not isinstance(data, Dataset):
            raise ValueError("Data parameter must be an instance of Dataset!")
        if data.data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        if precision is not None and not 0 <= precision <= 1:
            raise ValueError("Invalid precision value. Must be on interval [0,1]!")
        if precision is not None:
            kwargs["target_precision"] = precision
        self.data = data.data
        self.size = len(data.data)
        self.params = self.flann.build_index(data.data, algorithm="kdtree", trees=trees, **kwargs)
