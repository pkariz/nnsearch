from ..flannindex import FlannIndex
import numpy as np
from ..datasets import Dataset


class KDTree(FlannIndex):
    """KDTree algorithm from FLANN optimized for lower dimensions."""

    def __init__(self):
        super(KDTree, self).__init__()
        self.algorithm = "KD-Tree-flann"
        self.valid_types = [np.uint8,
                            np.int32,
                            np.float32, np.float64]

    def build(self, data, leaf_size=10, **kwargs):
        """Builds flann's KDTreeSingle index.
        :param data: Dataset instance representing data
        :param leaf_size: maximum number of points in a leaf, determines when the branching ends.
        """
        if not isinstance(data, Dataset):
            raise ValueError("Data parameter must be an instance of Dataset!")
        if data.data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        self.data = data.data
        self.size = len(data.data)
        self.params = self.flann.build_index(data.data, algorithm="kdtree_single", leaf_max_size=leaf_size, **kwargs)
