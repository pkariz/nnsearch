import numpy as np
import os


_dir = os.path.dirname(os.path.abspath(__file__))

_samples = {
    #"wiki": "wiki100000.npy",
    #"aerial" : "aerial.npy",
    #"disk_trace" : "disk_trace.npy",
    "german_post_codes" : "german_post_codes.npy",
    #"gisette" : "gisette_train.npy",
    #"amazon" : "amazon_50_30_10000.npy",
    "circle_2d": "circle_2d_100000.npy",
    "circle_3d": "circle_3d_100000.npy",
    "circle_5d": "circle_5d_100000.npy",
    "circle_10d": "circle_10d_100000.npy",
    "clusters_2d": "clusters_2d_10c_100000.npy",
    "clusters_3d": "clusters_3d_10c_100000.npy",
    "elipse_2d": "elipse10_04_2d_100000.npy",
    "elipse_3d": "elipse_3d_144_414_441_100000.npy",
    "dim1000": "dim1000_10000.npy",
    "uniform_1d_100000": "uniform_1d_100000.npy"
}

samples = _samples.keys()


def load_dataset(name):
    global _dir, _samples
    if name not in _samples:
        raise ValueError("Invalid dataset name")
    data = np.load(os.path.join(_dir,"sample/"+_samples[name]))
    return Dataset(name=name, data=data)


class Dataset(object):
    """Object which represents dataset."""
    def __init__(self, name="Dataset1", filename=None, sep=",", data=None, d=None, attr_names=None, dtype=None):
        """
        Initialized dataset instance from file (requires 'filename' and 'sep'), from data (requires 'data') or creates
        an empty dataset (requires 'd').

        :param name: Name of dataset
        :param filename: path to file
        :param sep: value separator used in given file
        :param data: 2d numpy array representing data
        :param d: number of dimensions of vectors
        :param attr_names: names of attributes, automatically generated if not specified
        :param dtype: numpy dtype information, default is np.float64
        """

        if dtype is None:
            dtype = np.float32
        self.name = name
        self.c = None #maximum coordinate used when transforming to binary
        self.normalized = {} #key = normalized column, value = denominator (values of this column were divided by this)
        if filename:
            self.attr_names, self.data = self._get_data_from_file(filename, sep, dtype)
        else:
            if data is not None:
                self.data = data
                self.attr_names = attr_names or Dataset._generate_names(len(data[0]))
            else:
                if d is None:
                    raise ValueError("You need to specify number of dimensions of vectors with keyword argument 'd'!")
                self.data = np.empty((0, d)) #empty array of d-dimensional vectors, dtype=float64
                self.attr_names = attr_names or Dataset._generate_names(d)

    @staticmethod
    def _is_number(s):
        """
        Returns True if parameter can be converted in float type.

        :param s: string
        :return: True if string can be converted to float, otherwise False
        """

        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _generate_names(n):
        """
        Generates names of n attributes and returns them in a list.

        :param n: number of names
        :return: list of attribute names
        """

        return ["x"+str(i) for i in range(1,n+1)]

    def _get_data_from_file(self, filename, sep, dtype):
        """Gets data from file. File can contain attribute names (which can't be cast to float numbers)
        in the first line followed by an empty line and vectors start in the third line. Otherwise vectors
        start in the first line and default attributes names are generated. All attribute values are
        separated by the separator.

        Example of a file:
        line1: x,y,z
        line2:
        line3: 3.10, 4.15, 1.1
        line4: 5.5, 10.0, 15.2

        :param filename: path to file
        :param sep: separator used in this file
        :param dtype: numpy dtype information
        :return list of attribute names and 2d numpy array representing vectors.
        """

        with open(filename,'r') as f:
            lines = f.readlines()
            first_line = lines[0].strip().split(sep)
            #check if the file contains attribute names
            attr_names = []
            if Dataset._is_number(first_line[0]):
                attr_names = Dataset._generate_names(len(first_line))
                start = 0
            else:
                attr_names = first_line
                start = 2
            #read vectors
            vectors = []
            for line in lines[start:]:
                vectors.append(map(float,line.strip().split(sep)))
        return attr_names, np.array(vectors, dtype=dtype)

    def add_vector(self, v):
        """
        Adds a vector to the dataset. Parameter can be tuple, list or numpy array.
        :param v: vector to add
        """

        self.data = np.vstack((self.data,np.array(v)))


    def _split_dataset(self, dataset, i):
        """
        Constructs two parts, one is Dataset object and the other one is a 2d numpy array.
        Parameters are dataset (which is Dataset.data split into k pieces) and integer i.
        The Dataset object contains all vectors from the dataset except those on i-th slot, which are
        returned in a 2d numpy array as queries.

        :param dataset: Dataset.data split into k pieces
        :param i: integer defining which piece of dataset is returned as queries
        :return: tuple consisting of Dataset object (containing data from all pieces except i-th) and 2d numpy
        array representing queries (i-th piece).
        """

        queries = dataset[i]
        others = np.delete(dataset,i)
        train_data = np.concatenate(others, axis=0)
        return Dataset(name="fold-"+str(i), data = train_data), queries

    def k_folds(self, k=2, p=None, random=False):
        """Generator for getting k split datasets.

        :param k: number of splits
        :param p: proportion of vectors in Dataset object, the rest are queries. If None then
            p = len(dataset)/k and dataset is permuted at most once
        :param random: if True, this method works on a random permutation of the dataset. If p
            is set then random is always True

        It yields one Dataset object (train vectors) and one 2d numpy array (query vectors).
        """

        if not 1<k<len(self.data) or (p is not None and not 0.0<p<1.0):
            print "Invalid parameters!"
            return
        d = self.data

        if p:
            for i in range(k):
                d = np.random.permutation(self.data) #random permutation of rows in dataset
                d = np.array_split(d,1.0/p) #split dataset in 1.0/p pieces
                new_data, queries = self._split_dataset(d, 0) #first piece = queries, others = train
                yield new_data, queries
        else:
            if random:
                d = np.random.permutation(self.data) #random permutation of rows in dataset
            d = np.array_split(d,k) #split in k pieces
            for i in range(k):
                new_data, queries = self._split_dataset(d, i)
                yield new_data, queries


    def normalize(self, cols=None):
        """
        Normalizes given columns by dividing value with np.linalg.norm(column) = L2-norm.

        :param cols: list of column indexes which need to be normalized, default is all columns.
        """

        if cols is None:
            cols = range(len(self.data[0]))

        self.data = self.data.T
        new_data = []
        for c in cols:
            minc, maxc = None, None
            #get min and max in dimension c
            for v in self.data[c]:
                if minc is None or v < minc:
                    minc = v
                if maxc is None or v > maxc:
                    maxc = v
            try:
                col = (self.data[c] - minc)/float(maxc - minc)
            except Exception, e:
                print "maxc:%f, minc:%f" % (maxc, minc)
                print self.data[c]
                raise e
            new_data.append(col)
            self.normalized[c] = (minc, maxc) #normalization (X-minc)/(maxc-minc)
        new_data = np.array(new_data)
        self.data = new_data.T


    def normalize_queries(self, queries):
        """
        Normalizes queries based on previous dataset normalization.

        :param queries: 2d numpy array representing query points
        :return: 2d numpy array with normalized queries
        """
        if len(self.normalized) == 0:
            raise Exception("Need to normalize data first!")
        transposed = queries.T
        new_queries = []
        for c, (minc, maxc) in self.normalized.items():
            new_queries.append([min(1.0,max(0.0, x)) for x in (transposed[c]-minc)/float((maxc-minc))])

        new_queries = np.array(new_queries)
        return new_queries.T

    @staticmethod
    def _binarize(data, c=None):
        if not c:
            c = np.amax(data)
        new_arr = []
        for i in range(len(data)):
            new_v = []
            for a in data[i]:
                binarized_attr = [1]*int(a)
                binarized_attr.extend([0]*(c-int(a)))
                new_v.extend(binarized_attr)
            new_arr.append(new_v)

        return np.array(new_arr, dtype=np.uint8)

    def to_binary(self, queries=None):
        """
        Transforms each vector in dataset to a binary vector with the help of unary representation. If queries are given
        it also transforms them into binary. When the function is called for the first time unary representation of
        vectors is derived from data and queries (if given). The new number of dimensions of Dataset.data and queries is
        C*d, where C is the maximum coordinate found when this function was called for the first time. Data must consist
        of integers only.

        :param queries: 2d numpy array representing query points which need to be transformed to binary
        :return: 2d numpy array with binary representation of query points or None if queries are not given
        """

        #find max coordinate
        if self.c is None:
            if queries is not None:
                self.c = max(np.amax(self.data), np.amax(queries))
                self.data = Dataset._binarize(self.data,self.c)
                return Dataset._binarize(queries, self.c)
            else:
                self.c = np.amax(self.data)
                self.data = Dataset._binarize(self.data,self.c)
                return None
        else:
            if queries is not None:
                return Dataset._binarize(queries, self.c)
            raise ValueError("No data to transform (dataset is already, queries were not given)!")


    def __repr__(self):
        return "%s(name=%r, data=%r)" % (self.__class__.__name__, self.name, self.data)

    def __str__(self):
        return "Dataset name: %s\nAttributes: %s\nNr. of vectors: %d" % (self.name, self.attr_names, len(self.data))



