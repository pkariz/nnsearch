import warnings
from Annoy import Annoy
from BF import BF as BoundaryF
from LSHNearPy import LSHNearPy
try:
    from pyflann import *
    from FlannAuto import FlannAuto
    from KMeans import KMeans as HKmeans
    from LSHFlann import LSHFlann
    from RKDTree import RKDTree
except:
    warnings.warn("Missing Flann library: FlannAuto, LSHFlann, HKmeans and RKDTree not imported!")
    


