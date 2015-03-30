import numpy as np
import random
from nnsearch.distances import minkowski


class Hypercube(object):
    """Class representing Hypercube, used in LiNearN"""
    def __init__(self):
        self.center = None
        self.radius = None
        self.mass = None


def density(data, p=-1, t=20, a=4, b=32):
    hs = _linearN(data, t, a, b, p)
    densities = []
    for i, x in enumerate(data):
        r = _density(x, hs, p)
        densities.append(r)
    densities = np.array(densities)
    return np.mean(densities), np.std(densities)


def _search(h, x, p):
    """Returns hypercube that covers x. If none of them covers x it returns None."""
    for hc in h:
        if minkowski(hc.center, x, p=p) <= hc.radius:
            return hc
    return None


def _build_hypercubes(data, p):
    """
    Builds hypercubes with L_inf norm.
    :param data: data points
    :return: list of hypercubes
    """
    def get_closest(x, idx, p):
        """Returns distance to the closest point to x, excluding index idx, which is the index of x."""
        closest_dist = None
        for i in range(0, len(data)):
            if i == idx:
                continue
            cur_dist = minkowski(data[i], x, p=p)
            if closest_dist is None or cur_dist < closest_dist:
                closest_dist = cur_dist
        return closest_dist

    h = []
    for m in range(0,len(data)):
        hc = Hypercube()
        hc.center = data[m]
        #brute-force closest with L_inf norm
        dist = get_closest(hc.center, m, p)
        hc.radius = 1/2.0 * dist
        hc.mass = 0
        h.append(hc)
    return h

def _assignSampleMass(hs, data, b, p):
    """Assigns estimated masses to hypercubes."""
    for i in range(len(hs)):
        di = random.sample(data, b)
        for j in range(b):
            hc = _search(hs[i], di[j], p)
            if hc is not None:
                hc.mass += 1


def _density(x, hs, p):
    """Returns average density estimated for point x."""
    res = 0.0
    for i in range(len(hs)):
        hc = _search(hs[i], x, p)
        if hc is not None:
            res += hc.mass/float(hc.radius)
    return res

def _linearN(data, t, a, b, p):
    """
    Performs LinearN algorithm with L_inf - norm.
    :param data: 2d numpy array with data points
    :param t: number of subsamples D
    :param a: the size of subsample used for constructing a set of hypercube regions
    :param b: the size of subsample used to estimate density in Hi
    :return: list of lists of HyperCube objects. Each element contains a total 'a' of non-overlapping hypercube regions
    with estimated densities.
    """
    hs = []
    for i in range(t):
        d = random.sample(data, a)
        hs.append(_build_hypercubes(d, p))

    _assignSampleMass(hs, data, b, p)
    return hs