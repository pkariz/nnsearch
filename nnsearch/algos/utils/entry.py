from rectangle import Rect
import numpy as np


class Entry(object):
    """Object representing Entry of a tree."""
    def __init__(self, e, e_mbr=None):
        #if mbr = None a point was given
        if not e_mbr:
            e_mbr = Rect(e, e)
        self.entry = np.array(e)
        self.mbr = e_mbr
        self.parent = None

    def __eq__(self, other):
        return (self.entry == other.entry).all()

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return str(self.entry)
