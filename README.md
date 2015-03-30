nnsearch
========

## What is nnsearch?
It's a python package for searching exact and approximate nearest neighbors. It contains implementations of R-tree, R*-tree, Pivoted M-tree and BoundaryForest. It also wraps some algorithms from other libraries and is simple to use.

## Prerequisites?
- numpy
- matplotlib
- FLANN: [FLANN page](http://www.cs.ubc.ca/research/flann/)installation is explained in their manual
- Annoy: pip install https://pypi.python.org/packages/source/a/annoy/annoy-1.0.5.tar.gz
       if you dont have boost then install boost first --> apt-get install libboost-all-dev
- NearPy: pip install NearPy
- scikit-learn: apt-get install python-sklearn
