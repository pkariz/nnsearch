from ..flannindex import FlannIndex
import numpy as np
from ..datasets import Dataset

class BruteForce(FlannIndex):
    """Brute-force algorithm from FLANN."""

    def __init__(self):
        super(BruteForce, self).__init__()
        self.algorithm = "Brute-force-flann"
        self.valid_types = [np.uint8,
                            np.int32,
                            np.float32, np.float64]

    def build(self, data, **kwargs):
        """Brute-force, doesnt actually build any index.
        :param data: Dataset instance representing data
        """
        if not isinstance(data, Dataset):
            raise ValueError("Data parameter must be an instance of Dataset!")
        if data.data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        self.data = data.data
        self.size = len(data.data)
        self.params = self.flann.build_index(data.data, algorithm="linear", **kwargs)

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError
