from ..baseindex import Index
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter
from nearpy.storage import MemoryStorage
import numpy as np
import cPickle
from ..datasets import Dataset

class LSHNearPy(Index):
    """
    LSH implementation from NearPy.
    """
    def __init__(self):
        self.algorithm="LSH-NearPy"
        self.valid_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64,
                            np.float16, np.float32, np.float64]

    def build(self, data=None, dimensions=None, lshashes=None,
              distance=None, vector_filters=None, storage=None):
        """
        Builds LSH based on lshashes.

        :param data: Dataset instance representing vectors to hash (if dimensions parameter is given then this is optional)
        :param dimensions: number of dimensions of data (if data is present then this parameter is optional)
        :param lshashes: hash functions from nearpy.hashes
        :param distance: distance function used, default is euclidean
        :param vector_filters: list of filters used on candidates, can be NearestFilter(k), DistanceThresholdFilter(d)
        and UniqueFilter()
        :param storage: defines where data is stored, either MemoryStorage or RedisStorage
        """

        if dimensions is None and data is None:
            raise ValueError("Either data or dimensions must be passed as parameters!")
        if not dimensions is None and data is not None and dimensions != len(data.data[0]):
            raise ValueError("Dimensions from parameter 'dimensions' and derived dimensions from 'data' are different!")
        dimensions = dimensions or len(data.data[0])
        if lshashes is None:
            lshashes = [RandomBinaryProjections('default', 10)]
        if distance is None:
            distance = EuclideanDistance()
        if vector_filters is None:
            vector_filters = [NearestFilter(1)]
        if storage is None:
            storage = MemoryStorage()

        #save current 'k'
        self.k = 1
        for x in vector_filters:
            if isinstance(x, NearestFilter):
                self.k = x.N
        self.index = Engine(dimensions, lshashes=lshashes, distance=distance, vector_filters=vector_filters,
                            storage=storage)
        self.size = 0
        if data is not None:
            if not isinstance(data, Dataset):
                raise ValueError("Data parameter must be an instance of Dataset!")
            if data.data.dtype not in self.valid_types:
                raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
            #insert data
            for v in data.data:
                self.insert(v)


    def insert(self, vector, data=None):
        """
        Stores vector in storage.
        :param vector: vector to store
        :param data: must be JSON-serializable, is also stored and returned when querying
        """
        self.index.store_vector(vector, data)
        self.size += 1

    #@profile
    def query(self, queries, k, return_data=False, **kwargs):
        """
        Finds k nearest neighbors of each query.

        :param queries: single query (1d numpy array) or multiple queries (2d numpy array)
        :param k: number of nearest neighbors (removes old NearestFilter and adds a new one)
        :param return_data: if True also data information of vectors will be returned
        :return: if there is only 1 query then it returns tuple (neighbors, distances) where neighbors is 2d numpy array
        with i-th element representing i-th nearest neighbour and distances is 1d numpy array of distances (i-th distance
        corresponds to i-th k-nn). If there are more queries returns are similar but each returned value has 1 dimension
        more (representing query). If return_data is True tuple will consist of 3 elements, the last one being data
        informations about each nearest neighbour returned.
        """
        if k != self.k:
            for i in self.index.vector_filters:
                if isinstance(i, NearestFilter):
                    self.index.vector_filters.remove(i)
            self.k = k
            self.index.vector_filters.append(NearestFilter(k))
        if len(queries.shape) == 1:
            res = self.index.neighbours(queries)
            neighbors, all_data, distances = [], [], []
            for vector, data, dist in res:
                neighbors.append(vector)
                all_data.append(data)
                distances.append(dist)
            if return_data:
                return np.array(neighbors), np.array(distances), all_data
            else:
                return np.array(neighbors), np.array(distances)
        else:
            neighbors, all_data, distances = [], [], []
            for q in queries:
                res = self.index.neighbours(q)
                cur_neighbors, cur_data, cur_dists = [], [], []
                for vector, data, dist in res:
                    cur_neighbors.append(vector)
                    cur_data.append(data)
                    cur_dists.append(dist)
                neighbors.append(cur_neighbors)
                all_data.append(cur_data)
                distances.append(cur_dists)
            if return_data:
                return np.array(neighbors), np.array(distances), all_data
            else:
                return np.array(neighbors), np.array(distances)

    def candidate_count(self, query):
        """
        Returns number of candidates from the same buckets as query.
        :param query: query point
        :return: number of candidates
        """

        return self.index.candidate_count(query)

    def clean_all_buckets(self):
        """Clears all buckets in storage."""
        self.index.clean_all_buckets()

    def clean_buckets(self, hash_name):
        """Clears all buckets of hashing function with name hash_name."""
        self.index.clean_buckets(hash_name)

    def save(self, filename):
        """
        Saves this instance to file.
        :param filename: file path
        """

        f = open(filename,"wb")
        info = {"index": self.index, "size": self.size, "k": self.k}
        cPickle.dump(info, f)
        f.close()

    def load(self, filename):
        """
        Loads instance from file.
        :param filename: file path
        """

        f = open(filename,"rb")
        info = cPickle.load(f)
        self.index = info["index"]
        self.size = info["size"]
        self.k = info["k"]
        f.close()
