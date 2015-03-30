from ..baseindex import Index
from ..algos import RTree as RTreeIndex
from ..algos.utils import Entry
import numpy as np
from ..datasets import Dataset


class RTree(Index):
    """Implementation of R-tree."""

    def __init__(self):
        self.algorithm = "R-Tree"
        self.valid_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64,
                            np.float16, np.float32, np.float64]

    def build(self, data=None, dimensions=None, min_node_size=None, max_node_size=100, method="linear"):
        """
        Builds R-tree from data. Because bulk-loading is not implemented it inserts points one by one.

        :param data: Dataset instance representing points. This parameter is optional, if None then empty tree is created
        :param dimensions: number of dimensions in the tree. This parameter is optional unless data is None
        :param min_node_size: minimum number of rectangles in a node, best if this is equal to max_nr_rects * 0.4
        :param max_node_size: maximum number of rectangles in a node, must be >= 2
        :param method: split method, can be "linear" or "quadratic"
        """
        if max_node_size < 2:
            raise ValueError("Maximum number of rectangles is too low, must be at least 2!")
        if min_node_size is None:
            min_node_size = max(1, int(max_node_size*0.4))
        if dimensions is None and data is None:
            raise ValueError("Either data or dimensions must be passed as parameters!")
        if not dimensions is None and not data is None and dimensions != len(data.data[0]):
            raise ValueError("Dimensions from parameter 'dimensions' and derived dimensions from 'data' are different!")
        if min_node_size >= max_node_size:
            raise ValueError("Minimum number of rectangles must be smaller than maximum!")
        dimensions = dimensions or len(data.data[0])

        self.index = RTreeIndex(dimensions, min_node_size, max_node_size, method)
        self.size = 0
        self.height = 0
        if data is not None and data.data is not None:
            if not isinstance(data, Dataset):
                raise ValueError("Data parameter must be an instance of Dataset!")
            if type(data.data) is np.ndarray and data.data.dtype not in self.valid_types:
                raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
            for x in data.data:
                self.insert(Entry(x))
            self.size = len(data.data)

    def insert(self, entry):
        """
        Inserts entry in the tree.
        :param entry: can be an instance of Entry or some iterable representing point.
        """

        if isinstance(entry,Entry):
            self.index.insert_data(entry)
        else:
            self.index.insert_data(Entry(entry))
        self.size += 1
        self.height = self.index.height

    def delete(self, entry):
        """
        Deletes entry from the tree if tree contains it.
        :param entry: entry
        :return: True if deleted otherwise False
        """

        res = self.index.delete(entry)
        self.size = self.index.size
        self.height = self.index.height
        return res

    def query(self, queries, k=1, mink_p=2):
        """
        Returns k nearest neighbors for each query and (optionaly) their distances.

        :param queries: 2d numpy array of queries
        :param k: number of nearest neighbors
        :param mink_p: defines which L_p norm to use as distance
        :return: tuple where first element is 2d numpy array of nearest neighbors (their Entry instances) and second is 2d numpy array of their
        distances (or None if return_distances is False)
        """

        res_neighbors = []
        res_distances = []
        if isinstance(queries, np.ndarray) and len(queries.shape) == 1 or \
                isinstance(queries, list) and not isinstance(queries[0], list):
            ns, ds = self.index.query(queries, k, mink_p)
            if isinstance(queries, np.ndarray):
                ns = np.array(ns)
                ds = np.array(ds)
            return ns, ds
        else:
            for q in queries:
                ns, ds = self.index.query(q, k, mink_p)
                if isinstance(queries, np.ndarray):
                    ns = np.array(ns)
                    ds = np.array(ds)
                res_neighbors.append(ns)
                res_distances.append(ds)
            if isinstance(queries, np.ndarray):
                res_neighbors = np.array(res_neighbors)
                res_distances = np.array(res_distances)
            return res_neighbors, res_distances

    def plot(self, filename=None, marker_size=2, height=None):
        """
        Plots R*-Tree, only available in 2d and 3d space. Root rectangle is never drawn so if height is set to 2
        only roots children rectangle will be drawn. Points are always drawn.

        :param filename: file in which the figure is saved (optional), eg. "my_figure.png"
        :param marker_size: size of marker
        :param height: up untill which height should the rectangles be drawn
        """

        self.index.plot_tree(filename=filename, marker_size=marker_size, height=height)

    def save(self, filename):
        """
        Saves R-Tree to file.
        :param filename: file in which R-Tree is saved. It should have ".cpickle" extension.
        """
        self.index.save(filename)

    def load(self, filename):
        """
        Loads R-Tree from file.
        :param filename: file from which R-Tree is loaded.
        """
        self.index = RTreeIndex(100).load(filename)
        self.size = self.index.size
        self.height = self.index.height