from ..baseindex import Index
import numpy as np
import math
from annoy import AnnoyIndex


class Annoy(Index):
    """
    AnnoyIndex from annoy package.
    """

    def __init__(self):
        self.algorithm = "AnnoyIndex"
        self.idx_to_vector = {}
        self.valid_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64,
                            np.float16, np.float32, np.float64]

    def build(self, data=None, dimensions=None, distance="angular", trees=-1):
        """
        Builds AnnoyIndex on data or creates an empty one. If both dimensions and data are given then their dimensions
        must match. At least one of those two attributes must be given to define number of dimensions which is required
        to create AnnoyIndex. After the trees are built you cannot add additional vectors.

        :param data: Dataset instance representing vectors which are inserted before trees are built (optional, you can
        insert data one by one with insert method before building trees)
        :param dimensions: number of dimensions
        :param distance: can be "angular" (default) or "euclidean"
        :param trees: number of binary trees. Default (-1) means that this parameter is determined automatically in a way,
        that memory usage <= 2 * memory(vectors)
        """

        #check dimensions
        if data is None and dimensions is None:
            raise ValueError("Number of dimensions is missing!")
        if data is not None and dimensions is not None and dimensions != len(data.data[0]):
            raise ValueError("Dimensions from constructor parameter 'dimensions' and derived dimensions from 'data' are different!")
        #build index
        if data is not None:
            dimensions = len(data.data[0])
        self.index = AnnoyIndex(dimensions, distance)
        self.d = dimensions
        self._size = 0
        self.metric = 0 #angular
        if distance != "angular":
            self.metric = 1 #euclidean

        #fill data
        if data is not None:
            if type(data.data) is np.ndarray and data.data.dtype not in self.valid_types:
                raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
            for v in data.data:
                self._insert(v)
        #build trees
        self.index.build(trees)

    def _insert(self, vector):
        """
        Inserts vector in AnnoyIndex.

        :param vector: 1d numpy array, list or tuple representing vector
        """
        if type(vector) is np.ndarray:
            vector = vector.tolist()
        else:
            vector = list(vector)
        self.index.add_item(self._size, vector)
        self._size += 1

    def get_dist(self, v1, v2, dist=None):
        """
        Calculates distance (euclidean or angular) between two vectors. By default distance is set to metric of index.
        :param v1: first vector (list or numpy array)
        :param v2: second vector
        :param dist: distance can be 0 (angular) or 1 (euclidean)
        :return: distance between given vectors
        """
        if dist is None:
            dist = self.metric
        if dist == 0:
            #angular
            v1_sum, v2_sum, mix_sum = 0.0, 0.0, 0.0
            for i in range(self.d):
                v1_sum += v1[i] * v1[i]
                v2_sum += v2[i] * v2[i]
                mix_sum += v1[i] * v2[i]
            a = v1_sum * v2_sum
            if a > 0.0:
                return 2.0 - (2.0 * mix_sum / (math.sqrt(a)))
            else:
                return 2.0
        else:
            #euclidean
            d = 0.0
            if self.d != len(v1) or self.d != len(v2):
                raise ValueError("Length of vectors is not the same as d!")
            for i in range(self.d):
                d += (v1[i] - v2[i]) * (v1[i] - v2[i])
            return math.sqrt(d)

    def query(self, queries, k=1):
        """
        Returns k nearest neighbors.
        :param queries: 1d or 2d numpy array or list
        :param k: number of nearest neighbors to return
        :return: array with k nearest neighbors, if return_distances is True it returns (a,b) where a is array with k
        nearest neighbors and b is an array with the same shape containing their distances
        """
        dists = []
        if isinstance(queries, np.ndarray) and len(queries.shape) == 1 or \
                isinstance(queries, list) and not isinstance(queries[0], list):
            if isinstance(queries, np.ndarray):
                neighbors = self.index.get_nns_by_vector(queries.tolist(), k)
            else:
                neighbors = self.index.get_nns_by_vector(queries, k)
            #calculate distances
            dists = [self.get_dist(queries.tolist(), self.index.get_item_vector(x)) for x in neighbors]
        else:
            #more queries
            neighbors = []
            for query in queries:
                if isinstance(query, np.ndarray):
                    cur_neighbors = self.index.get_nns_by_vector(query.tolist(), k)
                else:
                    cur_neighbors = self.index.get_nns_by_vector(query, k)
                neighbors.append(cur_neighbors)
                #calculate distances from cur_neighbors to query point
                dists.append([self.get_dist(query, self.index.get_item_vector(x)) for x in cur_neighbors])

        return np.array(neighbors), np.array(dists)


    def save(self, filename):
        """Saves index to file."""
        self.index.save(filename)

    def load(self, filename, dimensions=None, distance=None):
        """
        Loads index from file.
        :param filename: path to file
        :param dimensions: number of dimensions of index
        :param distance: distance used
        """
        if dimensions is None or distance is None:
            raise ValueError("Dimensions and distance are needed!")
        self.index = AnnoyIndex(dimensions, distance)
        self.d = dimensions
        self.metric = 0
        if distance == "euclidean":
            self.metric = 1
        self.index.load(filename)
