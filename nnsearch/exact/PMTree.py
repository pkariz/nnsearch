from ..baseindex import Index
from ..algos import PMTree as PMTreeIndex
from ..algos.pmtree import Entry
import numpy as np
from ..datasets import Dataset


class PMTree(Index):
    """Implementation of PM-tree."""

    def __init__(self):
        self.algorithm = "PM-Tree"
        self.valid_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64,
                            np.float16, np.float32, np.float64]

    def build(self, data=None, max_node_size=100, p=10, nhr=None, npd=None, distance="minkowski", mink_p=2,
              promote_fn="random", partition_fn="default", nr_pivot_groups=10):
        """
        Builds PM-tree from data. Since bulk-loading is not implemented it inserts points one by one.

        :param data: Dataset instance representing points. This parameter is optional, if None then empty tree is created
        :param max_node_size: maximum number of children in a node
        :param p: number of pivots
        :param nhr: number of pivots used in internal nodes
        :param npd: number of pivots used in leaves
        :param distance: can be "minkowski" or "edit_distance" or a custom function which computes distance between two
        objects
        :param mink_p: defines L_p norm when "minkowski" distance is used
        :param promote_fn: promote function used durring split. Can be "random", "mM_RAD" or custom function which
        accepts a list 'nodes' where you can access pivots through method get_pivot() (eg. nodes[0].get_pivot()). It
        must return two pivots which represent pivots of two nodes that are the result of a split.
        :param partition_fn: partition function used durring split. Can be "default" or custom function which accepts
        three parameters (o1, o2, nodes) where o1 and o2 are pivots returned by promote function, nodes is a list of
        nodes which contains also o1 and o2. It must return a tuple (group1, group2) where group1 and group2 are
        non-empty disjoint subsets of nodes.
        :param nr_pivot_groups: number of random pivot groups to construct from which the best one defines pivots.
        """
        if data is not None and len(data.data) <= p:
            raise ValueError("Number of pivots must not be greater than number of data points!")
        if nhr is not None and nhr > p or npd is not None and npd > p:
            raise ValueError("Parameters nhr and npd must not be greater than number of pivots!")
        if max_node_size < 2:
            raise ValueError("Maximum number of children must be at least 2!")
        if data is not None:
            if not isinstance(data, Dataset):
                raise ValueError("Data parameter must be an instance of Dataset!")
            if type(data.data) is np.ndarray and data.data.dtype not in self.valid_types:
                raise ValueError("Invalid dtype of numpy array, check valid_types parameter of index!")
            data = data.data
        self.index = PMTreeIndex(data, m=max_node_size, p=p, nhr=nhr, npd=npd, distance=distance, mink_p=mink_p,
                                 promote_fn=promote_fn, partition_fn=partition_fn, nr_pivot_groups=nr_pivot_groups)
        self.size = self.index.size
        self.height = self.index.height

    def insert(self, entry):
        """
        Inserts entry in the tree.
        :param entry: can be an instance of Entry or some iterable representing point.
        """
        if isinstance(entry, Entry):
            self.index.insert(entry)

        else:
            self.index.insert(Entry(entry))

        self.size += 1
        self.height = self.index.height



    def query(self, queries, k=1):
        """
        Returns k nearest neighbors for each query and (optionaly) their distances.

        :param queries: 2d numpy array of queries
        :param k: number of nearest neighbors
        :return: tuple where first element is 2d numpy array of nearest neighbors (their Entry instances) and second is 2d numpy array of their
        distances (or None if return_distances is False)
        """

        res_neighbors = []
        res_distances = []
        if isinstance(queries, np.ndarray) and len(queries.shape) == 1 or \
                        isinstance(queries, list) and not isinstance(queries[0], list):
            ns, ds = self.index.query(queries, k)
            if isinstance(queries, np.ndarray):
                ns = np.array(ns)
                ds = np.array(ds)
            return ns, ds
        elif isinstance(queries, np.ndarray) and len(queries.shape) == 2 or \
                        isinstance(queries, list) and isinstance(queries[0], list) and not isinstance(queries[0][0], list):#len(queries) == 2:
            for q in queries:
                ns, ds = self.index.query(q, k)
                if isinstance(queries, np.ndarray):
                    ns = np.array(ns)
                    ds = np.array(ds)
                res_neighbors.append(ns)
                res_distances.append(ds)
            if isinstance(queries, np.ndarray):
                res_neighbors = np.array(res_neighbors)
                res_distances = np.array(res_distances)
            return res_neighbors, res_distances
        elif isinstance(queries, basestring):
            ns, ds = self.index.query(queries, k)
            return ns, ds
        else:
            raise ValueError("Invalid query info!")

    def save(self, filename):
        """
        Saves PM-Tree to file.
        :param filename: file in which PM-Tree is saved. It should have ".cpickle" extension.
        """
        self.index.save(filename)

    def load(self, filename):
        """
        Loads PM-Tree from file.
        :param filename: file from which PM-Tree is loaded.
        """
        self.index = PMTreeIndex().load(filename)
        self.size = self.index.size
        self.height = self.index.height