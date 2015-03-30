import abc


class Index(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Creates an instance of Index."""

    @abc.abstractmethod
    def build(self, data, *args, **kwargs):
        """Builds index."""

    @abc.abstractmethod
    def query(self, queries, k, *args, **kwargs):
        """Returns nearest neighbors of queries."""

    @abc.abstractmethod
    def save(self, filename):
        """Saves index to file."""

    @abc.abstractmethod
    def load(self, filename, *args):
        """Loads index from file."""