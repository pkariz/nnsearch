from ..flannindex import FlannIndex
import numpy as np
from ..datasets import Dataset


class FlannAuto(FlannIndex):
    """Autotuned algorithm from FLANN."""

    def __init__(self):
        super(FlannAuto, self).__init__()
        self.algorithm = "FlannAuto"
        self.valid_types = [np.uint8,
                            np.int32,
                            np.float32, np.float64]

    def build(self, data, precision=0.9, **kwargs):
        """Builds flann's index based on autotuning algorithm.
        :param data: Dataset instance representing data
        :param precision: float between 0 and 1, higher value gives more accurate results. Default is 0.9.
        :param **kwargs: other flann's autotuning parameters, check flanns documentation
        """
        if not isinstance(data, Dataset):
            raise ValueError("Data parameter must be an instance of Dataset!")
        if data.data.dtype not in self.valid_types:
            raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
        self.data = data.data
        self.size = len(data.data)
        self.params = self.flann.build_index(data.data, algorithm="autotuned", target_precision=precision, **kwargs)
