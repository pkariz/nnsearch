from ..flannindex import FlannIndex
import numpy as np
from ..datasets import Dataset


class KMeans(FlannIndex):
    """Hierarchical KMeans algorithm from FLANN."""

    def __init__(self):
        super(KMeans,self).__init__()
        self.algorithm = "KMeans-flann"
        self.valid_types = [np.uint8,
                            np.int32,
                            np.float32, np.float64]

    def build(self, data, branching=32, iterations=5, centers_init="random", cb_index=0.5, precision=None, **kwargs):
        """Builds flann's kmeans index.
        :param data: Dataset instance representing data
        :param branching: branching factor to use for hierarchical kmeans tree creation
        :param iterations: maximum number of iterations when building kmeans tree. Value -1 means that clustering is
        performed until convergence
        :param centers_init: the algorithm to use for selecting the initial centers in kmeans clustering step. Possible
        values are 'random', 'gonzales' and 'kmeanspp'. More info in FLANN documentation.
        :param cb_index: this parameter (cluster boundary index) influences the way exploration is performed in the
        hierarchical kmeans tree. When cb index is zero the next kmeans domain to be explored is chosen to be the one
        with the closest center. A value greater then zero also takes into account the size of the domain.
        :param precision: the desired precision of searches
        """
        if not isinstance(data, Dataset):
            raise ValueError("Data parameter must be an instance of Dataset!")
        if data.data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        self.data = data.data
        self.size = len(data.data)
        if precision is not None and not 0 <= precision <= 1:
            raise ValueError("Invalid precision value. Must be on interval [0,1]!")
        if precision is not None:
            kwargs["target_precision"] = precision
        self.params = self.flann.build_index(data.data, algorithm="kmeans", branching=branching, iterations=iterations,
                                             centers_init=centers_init, cb_index=cb_index, **kwargs)
