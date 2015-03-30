import math
import multiprocessing
from itertools import imap
import random
import time
import logging
import sys
from multiprocessing import Manager, Process, Queue
from heapq import heappop, heappush, heapify
import cPickle
from functools import partial
import sys
import numpy as np

def mink2(x, y, p=2.0):
    """
    Returns minkowski distance between points x and y.
    :param x: first vector
    :param y: second vector
    :param p: defining Lp-norm, default is euclidean distance
    :return: distance between x and y using Lp-norm but not squared
    """
    #return math.sqrt(np.sum(np.power(abs(x-y), float(p))))
    res = 0.0
    for i in range(len(x)):
        #print "x[i]:%f, y[i]:%f, type:%s, cur_res:%f" %(x[i], y[i], type(abs(x[i]-y[i])), res)
        #print "x[i]:%f, y[i]:%f, type2:%s, cur_res:%f" %(x[i], y[i], type(max(x[i], y[i]) - min(x[i], y[i])), res)
        res += (max(x[i], y[i]) - min(x[i], y[i]))**p#abs(x[i]-y[i])#**p
    return math.sqrt(res)
    #return reduce(lambda acc, (idx, _): abs(x[idx]-y[idx])**p, enumerate(x), 0.0)


    #return math.sqrt(res)

tasks = {
    "knn":0,
    "classification":1,
    "regression":2
}

distances = {
    "euclidean": mink2
}

def chunks(seq, p):
    """ Chunks list which is a range of number [0...x] in 'p' parts
    :param seq: list
    :param p: number of parts
    :return: new list containing all the parts
    """
    newseq = []
    n = len(seq) / p #minimum per one
    r = len(seq) % p #others
    b,e = 0, n + min(1, r)  # first split
    for i in range(p):
        newseq.append(seq[b:e])
        r = max(0, r-1)  # use up remainders
        b,e = e, e + n + min(1, r)  # min(1,r) is always 0 or 1
    return newseq

def inverse_distance_weight(l, p=1):
    #l = [(node1, dist_to_query),(node2, dist_to_query)...]
    res = 0.0
    a = 0.0
    b = 0.0
    for n, d in l:
        if d == 0.0:
            return n.c
        a += n.c / float(d)
        b += 1 / (math.pow(float(d), p))
    return a / b


def same(bt1, bt2):
    if not all(bt1.x == bt2.x) or len(bt1.children) != len(bt2.children):
        return False
    for i in range(len(bt1.children)):
        if not same(bt1.children[i], bt2.children[i]):
            return False
    return True

class Node(object):

    def __init__(self, i):
        self.idx = i
        #self.c = [c[dim] for dim in dimensions]
        #self.original_x = x
        #self.original_c = c
        self.children = []

    def add_child(self, node):
        self.children.append(node)

class BoundaryTree(object):

    def __init__(self, i, data, labels, max_node_size, d, dc, task, eps, dimensions):
        self.root = Node(i)
        self.max_node_size = max_node_size
        self.d = d
        self.dc = dc
        self.task = task
        self.eps = eps
        self.size = 1
        self.dimensions = dimensions
        #self.vectors = data
        self.vectors = data[:,dimensions]
        self.labels = labels
        #self.vectors = data

    #@profile
    def train(self, i):
        #print "-------------inserting:%s---------------" % (x,)
        #xx = [x[d] for d in self.dimensions]#keep only relevant dimensions
        #cc = [c[d] for d in self.dimensions]
        v_min, _ = self.query(i, k=1)
        #res = self.query_knn(data, labels, i, k=1)
        #v_min = res[0][1] #closest node

        #print "query eturned point:", v_min
        #print "v_min.x:", v_min.x
        if self.task == 0:
            #knn, always add example
            new_node = Node(i)
            v_min.add_child(new_node)
            self.size += 1
        elif self.task == 1:
            #classification
            if self.labels[i] != self.labels[v_min.idx]:
                new_node = Node(i)
                v_min.add_child(new_node)
                self.size += 1
        else:
            #regression
            if abs(self.labels[v_min.idx] - self.labels[i]) > self.eps:
                new_node = Node(i)
                v_min.add_child(new_node)
                self.size += 1

    #@profile
    def query(self, i, k):
        v = self.root
        cur_node_dist = None
        while True:
            cur_best_node = None
            cur_smallest_dist = float("inf")
            for child in v.children:
                cur_d = self.d(self.vectors[child.idx], self.vectors[i])
                if cur_d < cur_smallest_dist:
                    cur_best_node = child
                    cur_smallest_dist = cur_d
            #check current node if it is not full
            if len(v.children) < self.max_node_size:
                if cur_node_dist is None:
                    cur_node_dist = self.d(self.vectors[v.idx], self.vectors[i])
                if cur_node_dist < cur_smallest_dist:
                    cur_best_node = v
                    cur_smallest_dist = cur_node_dist
            if cur_best_node == v:
                break
            v = cur_best_node
            cur_node_dist = cur_smallest_dist
        return v, cur_smallest_dist

    #@profile
    def query_knn(self, x, k):
        x = [x[d] for d in self.dimensions]#keep only relevant dimensions
        #start = time.clock()
        v = self.root
        cur_node_dist = self.d(self.vectors[v.idx], x)
        nn_heap = [(float("-inf"), False) for _ in range(k-1)]
        heappush(nn_heap, (cur_node_dist*-1.0, v))
        q = [] #heap with nodes to search
        while True:
            #cur_best_node = None
            #cur_smallest_dist = float("inf")
            for child in v.children:
                #print "len(child vec):%d, len(x):%d" % (len(self.vectors[child.idx]), len(x))
                cur_d = self.d(self.vectors[child.idx], x)
                heappush(q, (cur_d, child))
                if cur_d < nn_heap[0][0] * -1.0:
                    heappop(nn_heap)
                    heappush(nn_heap, (cur_d*-1.0, child))
                """if cur_d < cur_smallest_dist:
                    cur_best_node = child
                    cur_smallest_dist = cur_d"""
            """#check current node if it is not full
            if len(v.children) < self.max_node_size:
                #if cur_node_dist is None:
                    #cur_node_dist = self.d(v.x, x)
                if cur_node_dist < cur_smallest_dist:
                    cur_best_node = v
                    cur_smallest_dist = cur_node_dist"""
            """if cur_best_node == v:
                break
            if cur_best_node is None:
                break
            v = cur_best_node
            cur_node_dist = cur_smallest_dist"""
            if not q or q[0][0] > nn_heap[0][0] * -1.0:
                break
            v = q[0][1]
            cur_node_dist = q[0][0]
            heappop(q)
        #return neighbors
        return nn_heap

    def query_knn2(self, x, k):
        x = [x[d] for d in self.dimensions]#keep only relevant dimensions
        #start = time.clock()
        v = self.root
        cur_node_dist = self.d(self.vectors[v.idx], x)
        nn_heap = [(float("-inf"), False) for _ in range(k-1)]
        heappush(nn_heap, (cur_node_dist*-1.0, v))
        while True:
            cur_best_node = None
            cur_smallest_dist = float("inf")
            for child in v.children:
                cur_d = self.d(self.vectors[child.idx], x)
                if cur_d < nn_heap[0][0] * -1.0:
                    heappop(nn_heap)
                    heappush(nn_heap, (cur_d*-1.0, child))
                if cur_d < cur_smallest_dist:
                    cur_best_node = child
                    cur_smallest_dist = cur_d
                """if cur_d < nn_heap[0][0]*-1.0:
                    heappop(nn_heap) #odstrani prejsnjega
                    #print "nn_heap pred pushom:", nn_heap
                    #print "mislim pushat:", (cur_d*-1.0, child)
                    heappush(nn_heap, (cur_d*-1.0, child)) #das novega noter"""
            #check current node if it is not full
            if len(v.children) < self.max_node_size:
                #if cur_node_dist is None:
                    #cur_node_dist = self.d(v.x, x)
                if cur_node_dist < cur_smallest_dist:
                    cur_best_node = v
                    cur_smallest_dist = cur_node_dist
            if cur_best_node == v:
                break
            if cur_best_node is None:
                break
            v = cur_best_node
            cur_node_dist = cur_smallest_dist
        #return neighbors
        """neighbors = []
        distances = []
        while nn_heap:
            neighbor = heappop(nn_heap)
            neighbors.insert(0, neighbor[1])
            distances.insert(0, abs(neighbor[0]))#*-1.0)
        return neighbors, distances"""
        """end = time.clock() - start
        with open("query_knn_time.txt","w") as f:
            f.write("time:%f\n" % (end,))"""
        return nn_heap

    def get_v(self, v):
        if len(self.dimension) < len(v):
            return v
        return [v[i] for i in self.dimensions]

    def print_tree(self, node=None, level=0):
        """
        Prints tree in an awkward way which is good enough for debugging.
        :param node: node whose subtree is printed
        :param level: integer to know the current height from the starting node
        """
        if node is None:
            node = self.root
            print "-----------printing tree------------"
            if self.root is None:
                print "Tree is empty!"
                return
        s = ""
        if len(node.children) == 0:#leaf
            s = str(node.x)
            print '\t' * level + s
        else:
            print '\t' * level + str(node.x)
            for child in node.children:
                self.print_tree(child,level+1)

    def get_height(self, node=None, level=1):
        if node is None:
            node = self.root
            if node is None:
                return 0
        if len(node.children) == 0:
            return level
        h = level
        for child in node.children:
            h = max(h, self.get_height(child, level+1))
        return h

class BoundaryForest(object):
    """Boundary Forest implementation"""

    #@profile
    def __init__(self, data, labels=None, trees=4, max_node_size=50, task="knn", d="euclidean", dc="euclidean",
                 eps=None, parallel=True, n=5):
        """
        Builds boundary trees with specified parameters.
        :param data: 2d numpy array, size of data must be >= number of trees
        :param labels: labels of data, not needed for knn
        :param trees: number of boundary trees
        :param max_node_size: node capacity
        :param task: can be "knn", "regression" or "classification"
        :param d: distance used between two positions, in case of "knn" task positions are also labels
        :param dc: distance used between two labels, in case of "knn" task this is ignored since 'd' is used
        :param eps: defines an error window for regression problems. If a query result is > eps away from true value in
        a boundary tree then the estimate was wrong and it will create a new node with this 'missed' example in this
        tree.
        :param n: number of random dimensions to include for each tree, default is 5 (makes computations faster in
        higher dimensions).
        """
        #checks
        if task != "knn":
            raise NotImplementedError
        if data is None or not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise ValueError("Invalid data!")
        if not task in tasks:
            raise ValueError("Invalid task, possible tasks are \"knn\", \"regression\" and \"classification\"!")
        if task != "knn" and labels is None:
            raise ValueError("Labels missing!")
        if task != "knn" and len(data) != len(labels):
            raise ValueError("Lengths of data and labels must be the same!")
        if len(data) < trees:
            raise ValueError("Number of starting examples must be >= number of trees!")
        if task != "knn" and labels is None:
            raise ValueError("Labels are needed in classification and regression problems!")
        if trees < 1:
            raise ValueError("Number of trees must be at least 1!")
        if n > len(data[0]):
            n = len(data[0])
        self.trees = trees
        self.parallel = parallel
        self.max_node_size = max_node_size
        self.task = tasks[task]
        self.d = d
        self.dc = dc
        self.eps = eps
        self.bt = [] #list of boundary trees
        self.dist_x = None
        self.dist_c = None
        self.procs = [] #multiprocessing processes
        self.n = n
        self.size = len(data)
        self.data = data.copy() #we let user change the arrays
        self.labels = labels
        self.have_labels = False
        if self.labels is not None:
            self.have_labels = True
            self.labels = labels.copy()
        #check position distance
        if isinstance(self.d, basestring):
            if self.d in distances:
                self.dist_x = distances[d]
            else:
                raise ValueError("Invalid built-in distance!")
        elif hasattr(self.d, '__call__'):
            self.dist_x = d
        else:
            raise ValueError("Invalid distance argument, must be a function or one of allowed strings!")
        #check label distance if not knn
        if self.task != 0:
            if isinstance(self.dc, basestring):
                if self.dc in distances:
                    self.dist_c = distances[dc]
                else:
                    raise ValueError("Invalid built-in distance!")
            elif hasattr(self.dc, '__call__'):
                self.dist_c = dc
            else:
                raise ValueError("Invalid distance argument, must be a function or one of allowed strings!")
        else:
            self.dist_c = self.dist_x #knn uses positions as labels so the same distance is used
            labels = data

    def create_trees(self):
        #create roots
        if not self.parallel:
            for i in range(self.trees):
                dimensions = random.sample(range(len(self.data[0])), self.n) #choose random n dimensions for each tree
                #create a new tree with root = i-th data
                self.bt.append(BoundaryTree(i, self.data, self.labels, self.max_node_size, self.dist_x,
                                            self.dist_c, self.task, self.eps, dimensions))

                for j in range(len(self.data)): #add all other examples,  data >= trees
                    if i != j:
                        self.bt[i].train(j) #train tree on other data"
                    #if j%10000 == 0 and j!= 0:
                        #print "vstavil v to drevo:",j
                #print "finishal drevo:", str(i+1)
        else:
            #train on other examples parallel
            def create_trees_parallel(max_node_size, dist_x, dist_c, task, eps, n, idxs, out_q):
                new_bts = []
                for i, idx in enumerate(idxs):
                    dimensions = random.sample(range(len(self.data[0])), n) #choose random n dimensions for each tree
                    new_bts.append(BoundaryTree(idx, self.data, self.labels, max_node_size, dist_x,
                                            dist_c, task, eps, dimensions))
                    for j in range(len(self.data)): #add all other examples,  data >= trees
                        if idx != j:
                            new_bts[i].train(j) #train tree on other data"

                out_q.put(new_bts)

            out_q = Queue()
            self.procs = []
            cpus = multiprocessing.cpu_count()
            parts = chunks(range(self.trees), cpus) #split range of trees depending on number of cpu's
            parts = filter(lambda y: len(y) > 0, parts)
            partial_fn = partial(create_trees_parallel, self.max_node_size, self.dist_x,
                                 self.dist_c, self.task, self.eps, self.n)
            #PROCESS
            for i in range(cpus):
                proc = multiprocessing.Process(
                    target=partial_fn,
                    args = (parts[i], out_q))
                self.procs.append(proc)
                proc.start()
            res = []
            for _ in range(len(self.procs)):
                res.extend(out_q.get())
            for p in self.procs:
                p.join()
            self.procs = []
            correct_size = len(self.data)
            self.bt = res
            for i in range(self.trees):
                if self.bt[i].size != correct_size:
                    print "wtf, napacen size:", self.bt[i].size
                    print "mogu bi bit:", correct_size
                    exit(0)

    def train(self, x, c=None):
        #TODO: chunki train ne pa vsako drevo posebi proces
        i = self.size + 1
        """if self.parallel:
            def train_bts(bts, i):
                btrees = []
                for bt in bts:
                    bt.train()
            p = multiprocessing.Pool()
            cpus = multiprocessing.cpu_count()
            parts = chunks(range(self.trees), cpus) #split range of trees depending on number of cpu's
            parts = filter(lambda y: len(y) > 0, parts)
            btrees = p.map(train_bt, [([self.bt[idx] for idx in part], x, c) for part in parts] )
            self.bt = btrees
        else:"""
        self.data = np.vstack([self.data, x])

        for i in range(self.trees):
            self.bt[i].train(x, c)
        self.size += 1

    def get_queries(self, l, i, x, k):
        nearest, dist = BoundaryTree.get_query(self.bt[i], x, k)#btree.query(x, k)
        l[i] = (nearest, dist)
        #return nearest, dist

    #@profile
    def query(self, x, k=1, parallel=None):
        if len(x.shape) == 1:
            return self.query_point(x, k, parallel)
        elif len(x.shape) == 2:
            def query_pts(idxs, k, out_q):
                res = []
                for i in idxs:
                    ns, ds = self.query_point(x[i], k, False)
                    res.append((i, ns, ds))
                out_q.put(res)


            cpus = multiprocessing.cpu_count()
            parts = chunks(range(len(x)), cpus) #split range of trees depending on number of cpu's
            parts = filter(lambda y: len(y) > 0, parts)
            out_q = Queue()
            procs = []
            res = []
            #partial_fn = partial(sel, self.bt, x, k)
            for i in range(len(parts)):
                proc = multiprocessing.Process(
                    target=query_pts,
                    args=(parts[i], k, out_q))
                procs.append(proc)
                proc.start()
            for _ in range(len(procs)):
                res.append(out_q.get())
            for p in procs:
                p.join()
            neighbors = []
            distances = []
            for i, n, d in sorted([y for x in res for y in x]):
                neighbors.append(n)
                distances.append(d)
            return np.array(neighbors), np.array(distances)
        else:
            raise ValueError("Invalid query data!")

    #@profile
    def query_point(self, x, k=1, parallel=None):
        if parallel is None:
            parallel = self.parallel
        def get_queries4(bts, x, k, idxs, out_q):
            start = time.clock()
            res = []
            for i in idxs:
                start_inner = time.clock()
                nn_h = bts[i].query_knn(x, k)
                res.append(nn_h)
            candidates = {}
            nn_heap = []
            for l in res:
                for d, v in l:
                    if not isinstance(v, bool) and not v.idx in candidates:
                        candidates[v.idx] = True
                        heappush(nn_heap, (d, v))
                        if len(nn_heap) > k:
                            heappop(nn_heap)
            out_q.put(nn_heap)

        res = []
        moj_time = time.time()
        if parallel:
            cpus = multiprocessing.cpu_count()
            parts = chunks(range(self.trees), cpus) #split range of trees depending on number of cpu's
            parts = filter(lambda y: len(y) > 0, parts)
            out_q = Queue()
            procs = []
            partial_fn = partial(get_queries4, self.bt, x, k)
            for i in range(len(parts)):
                proc = multiprocessing.Process(
                    target=partial_fn,
                    args=(parts[i], out_q))
                procs.append(proc)
                proc.start()
            for _ in range(len(procs)):
                res.append(out_q.get())
            for p in procs:
                p.join()
        else:
            for i in range(self.trees):
                h = self.bt[i].query_knn(x, k)
                res.append(h) #append heap with knn

        if self.task == 0:
            candidates = {}
            nn_heap = []
            for l in res:
                for d,v in l:
                    if not isinstance(v, bool) and not v.idx in candidates:
                        candidates[v.idx] = True
                        heappush(nn_heap, (-d, v))
            neighbors = []
            distances = []
            cur_k = 0
            while nn_heap and cur_k != k:
                neighbor = heappop(nn_heap)
                if neighbor[0] == float("-inf"):
                    continue
                neighbors.append(self.data[neighbor[1].idx])
                #neighbors.append(neighbor[1].idx)
                distances.append(abs(neighbor[0]))
                cur_k += 1
            return np.array(neighbors), np.array(distances)

        elif self.task == 1:
            #classification
            labels = set([node.c for node, _ in res])
            labels_res = {x: [0.0, 0.0] for x in labels}
            for v, d in res:
                if d == 0:
                    return v.c
                labels_res[v.c][0] += v.c / float(d)
                labels_res[v.c][1] += 1 / float(d)
            best_label = None
            best_score = float("-inf")
            for label, info in labels_res.items():
                label_score = info[0] / info[1]
                if label_score > best_score:
                    best_label = label
                    best_score = label_score
            return best_label
        else:
            #regression
            return inverse_distance_weight(res)

    #@profile
    def query2(self, x, k=1):
        res = []
        for i in range(self.trees):
            vi, vi_dist = self.bt[i].query(x, k)
            res.append((vi, vi_dist))
        if self.task == 0:
            #locally closest node to x
            closest_node = None
            closest_dist = float("inf")
            for v, d in res:
                if d < closest_dist:
                    closest_node = v
                    closest_dist = d
            return closest_node, closest_dist
        elif self.task == 1:
            #classification
            labels = set([node.c for node, _ in res])
            labels_res = {x: [0.0, 0.0] for x in labels}
            for v, d in res:
                if d == 0:
                    return v.c
                labels_res[v.c][0] += v.c / float(d)
                labels_res[v.c][1] += 1 / float(d)
            best_label = None
            best_score = float("-inf")
            for label, info in labels_res.items():
                label_score = info[0] / info[1]
                if label_score > best_score:
                    best_label = label
                    best_score = label_score
            return best_label
        else:
            #regression
            return inverse_distance_weight(res)


    def save(self, filename):
        """
        Saves Boundary Forest in a specified file.

        :param filename: file in which the index is saved. It should have ".cpickle" extension.
        """
        #remove data and labels
        self.data = None
        self.labels = None
        #need to remove functions from attributes to pickle
        self.dist_x = None
        self.dist_c = None
        if not isinstance(self.d, basestring):
            #function was given
            self.d = None
        if not isinstance(self.dc, basestring):
            self.dc = None
        self.p = None
        for i in range(self.trees):
            #remove functions from all trees
            self.bt[i].d = None
            self.bt[i].dc = None
            #remove data from all trees
            self.bt[i].vectors = None
            self.bt[i].labels = None
        #sys.setrecursionlimit(500000000)
        """s = time.clock()
        f = open(filename,"wb")
        cPickle.dump(self, f, protocol=0)
        f.close()
        print "finished dumping protocol 0, time needed:", (time.clock()-s)"""

        s2 = time.clock()
        f = open(filename,"wb")
        cPickle.dump(self, f, protocol=2)
        f.close()
        #print "finished dumping protocol 2, time needed:", (time.clock()-s2)

    def load(self, filename, data, labels=None, d=None, dc=None, *args, **kwargs):
        """
        Loads Boundary Forest from file.

        :param filename: file from which Boundary Forest is loaded.
        :param data: data used to build the tree
        :param labels: labels used to build tree, can be None
        :param d: distance used between two positions, in case of "knn" task positions are also labels. Not needed if
        predefined function has been used.
        :param dc: distance used between two labels, in case of "knn" task this is ignored since 'd' is used. Not needed
         if predefined function has been used.
        :return: BoundaryForest instance
        """
        f = open(filename, "rb")
        self = cPickle.load(f)
        f.close()
        if self.have_labels == True and labels is None:
            raise ValueError("Labels are missing!")
        #setting right distance 'd' function
        if self.d is None:
            if d is None:
                raise ValueError("Distance function 'd' is needed!")
            self.dist_x = d
            self.d = d
        else:
            self.dist_x = distances[self.d]
        #setting right distance 'dc' function
        if self.dc is None:
            if dc is None:
                raise ValueError("Distance function 'dc' is needed!")
            self.dist_c = dc
            self.dc = dc
        else:
            self.dist_c = distances[self.dc]
        for i in range(self.trees):
            self.bt[i].d = self.dist_x
            self.bt[i].dc = self.dist_c
            #set vectors and labels for trees
            self.bt[i].vectors = data[:,self.bt[i].dimensions]
            self.bt[i].labels = labels
        self.data = data
        self.labels = labels
        return self
