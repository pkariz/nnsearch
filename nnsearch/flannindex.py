from baseindex import Index
from pyflann import *
import abc
import numpy as np

class FlannIndex(Index):
    """Represents flann index"""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Initialize flann instance."""
        self.flann = FLANN()
        self.params = None

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
                try:
                    #res = np.array(indices[0])
                    res = np.array([self.data[i] for i in indices[0]])
                except:
                    print "len od data:", len(self.data)
                    print "indices:", indices
                    print "queries:", queries
                    print "queries dtype:", queries.dtype
                    print "dists:", dists
                    exit(0)
                dists = np.array(dists[0])
            else:
                res = np.array([[self.data[i] for i in query] for query in indices])
                #res = np.array(indices)
                dists = np.array(dists)
        else:
            if isinstance(queries, np.ndarray) and len(queries.shape) == 1:
                dists = np.array(dists)
                #res = np.array(indices)
                res = np.array([self.data[i] for i in indices])
            else:
                res = np.array([np.array([self.data[i]]) for i in indices])
                #res = np.array([np.array([i]) for i in indices])
                dists = np.array([np.array([x]) for x in dists])
        return res, dists

    def save(self, filename):
        """Saves index to a file, the dataset is not saved."""
        self.flann.save_index(filename)

    def load(self, filename, data):
        """Loads index from file, dataset must be provided because index is saved without data."""
        self.flann.load_index(filename, data.data)
        self.size = len(data.data)
        self.data = data.data

    def set_distance(self, distance, order=0):
        """Sets distance to use when computing distances between points.

        :param distance: possible values are "euclidean", "manhattan", "minkowski", "max_dist", "hik", "hellinger",
        "cs", "kl". L_infinity is not valid, more info is available in flann manual
        :param order: only needed when distance is minkowski. Represents the order of minkowski distance.
        """
        set_distance_type(distance, order) #TODO: cekiraj ce ne to spremeni tudi ostalim indexom distance-a?
