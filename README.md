nnsearch
========

## What is nnsearch?
It's a python package for searching exact and approximate nearest neighbors. It contains implementations of R-tree, R*-tree, Pivoted M-tree and BoundaryForest. It also wraps some algorithms from other libraries and is simple to use.

##installing
First install prerequisites:
```bash
pip install numpy
pip install annoy
```
Install scipy: [scipy install page](http://www.scipy.org/install.html)
Install Flann: [FLANN page](http://www.cs.ubc.ca/research/flann/) (installation is explained in their manual)

You can then install package:
```bash
pip install nnsearch
```

Additional info: Annoy installation on windows might not work, installation of Flann takes time (it doesn't freeze as you might think).