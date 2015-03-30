from ..flannindex import FlannIndex
from ..datasets import Dataset
import numpy as np

class LSHFlann(FlannIndex):
    """Locality-sensitive hashing algorithm from FLANN."""

    def __init__(self):
        super(LSHFlann,self).__init__()
        self.algorithm = "LSH-flann"
        self.valid_types = [np.uint8]

    def build(self, data, nr_tables=12, key_size=20, multi_probe_level=2, precision=None, **kwargs):
        """Builds flann's locality-sensitive hashing index. Works only in Hamming space for binary descriptors.
        :param data: Dataset instance representing data
        :param nr_tables: number of hash tables
        :param key_size: the length of the key in the hash tables
        :param multi_probe_level: number of levels to use in multi-probe (0 for standard LSH)
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
        self.params = self.flann.build_index(data.data, algorithm="lsh", nr_tables=nr_tables, key_size=key_size,
                                             multi_probe_level_=multi_probe_level, **kwargs)

    def query(self, queries, k=1, **kwargs):
        """Searches for k nearest neighbors of each query point in queries.

        :param queries: 2d numpy array representing query points
        :param k: number of nearest neighbors
        :param **kwargs: other flann parameter info, eg: checks=32 which determines the number of times the index trees should be recursively searched, higher = slower & more precise

        :returns: sorted numpy array of nearest neighbors and numpy array with distances.
        """

        indices, dists = self.flann.nn_index(queries, k, **kwargs)
        if k != 1:
            if isinstance(queries, np.ndarray) and len(queries.shape) == 1:
                found = 1
                while found < len(dists[0]) and dists[0][found] >= 1.0 \
                    and indices[0][found] < len(self.data) and indices[0][found] > 0.0:
                    found += 1
                try:
                    res = np.array([self.data[i] for idx, i in enumerate(indices[0]) if idx < found])
                except:
                    print "len od data1:", len(self.data)
                    print "indices:", indices
                    print "queries:", queries
                    print "queries dtype:", queries.dtype
                    print "dists:", dists
                    exit(0)
                dists = np.array(dists[0][:found])
            else:
                new_res = []
                new_dists = []
                for query_idx in range(len(queries)):
                    found = 1
                    try:
                        while found < len(dists[query_idx]) and dists[query_idx][found] >= 1.0\
                                and indices[query_idx][found] < len(self.data) and indices[query_idx][found] > 0.0:
                            found += 1
                    except:
                        print "dists:", dists
                        print "query_idx:", query_idx
                        print "found:", found
                        exit(0)
                    new_res.append([self.data[i] for idx, i in enumerate(indices[query_idx]) if idx < found])
                    new_dists.append(dists[query_idx][:found])
                res = np.array(new_res)
                dists = np.array(new_dists)
        else:
            if isinstance(queries, np.ndarray) and len(queries.shape) == 1:
                found = 1
                dists = np.array(dists[:found])
                res = np.array([self.data[i] for i in indices[:found]])
            else:
                new_res = []
                new_dists = []
                for query_idx in range(len(queries)):
                    found = 1
                    while found < len(dists[query_idx]) and dists[query_idx][found] >= 1.0\
                            and indices[query_idx][found] < len(self.data) and indices[query_idx][found] > 0.0:
                        found += 1
                    new_res.append([self.data[i] for idx, i in enumerate(indices[query_idx]) if idx < found])
                    new_dists.append([dists[query_idx][:found]])
                res = np.array(new_res)
                dists = np.array(new_dists)
        return res, dists
