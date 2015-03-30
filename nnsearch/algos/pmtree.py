import numpy as np
import cPickle
import random
from heapq import heappop, heappush, heapify #storing k-nn
from nnsearch.distances import minkowski, edit_distance


class Node(object):
    """Represents node in PM-tree"""
    def __init__(self, node_pivot, radius, parent_dist, children, is_leaf, hr=None, nhr=None):
        self.pivot = node_pivot
        self.radius = radius
        self.children = children
        self.parent_dist = parent_dist
        self.parent = None
        self.is_leaf = is_leaf
        self.hr = hr #array of intervals defining ring regions eg. hr = [(min1, max1), (min2, max2)...]
        if hr is None:
            self.hr = [[float('inf'), float('-inf')] for i in range(nhr)] #infinity
        self.rand = random.random()

    def __eq__(self, other):
        return self.rand == other.rand

    def __ne__(self, other):
        return not self == other

    def update_hr(self, pmtree, e):
        """
        Updates intervals defining nodes rings depending on entry/node e.
        :param e: node/entry
        """
        for i in range(len(self.hr)):
            if isinstance(e, Entry):
                if i < len(e.pd):
                    #have distance from this entry to i-th pivot
                    dl = e.pd[i]
                else:
                    #need to calculate distance
                    dl = pmtree.distance_fn(e.entry, pmtree.pivots[i], **pmtree.distance_params)
                dr = dl
            else:
                #hr union
                dl = e.hr[i][0] #min
                dr = e.hr[i][1] #max

            if self.hr[i][0] > dl: #update min
                self.hr[i][0] = dl
            if self.hr[i][1] < dr: #update max
                self.hr[i][1] = dr

    def get_pivot(self):
        return self.pivot

    def update_radius(self, child, is_leaf):
        """
        Updates radius of a node depending on node's child.
        :param child: child of node
        """
        if is_leaf:
            self.radius = max(self.radius, child.parent_dist)
        else:
            self.radius = max(self.radius, child.parent_dist + child.radius)


    def add_children(self, pmtree, children):
        """
        Adds children from list 'children' to node.
        :param pmtree: PM-tree instance
        :param children: iterable containing children to add
        """

        for child in children:
            self.children.append(child)
            child.parent = self
            child.parent_dist = pmtree.distance_fn(child.get_pivot(), self.pivot, **pmtree.distance_params)
            self.update_radius(child, self.is_leaf)
            self.update_hr(pmtree, child)


    def add_entry(self, pmtree, e, distance):
        #radius has already been updated
        self.children.append(e)
        e.parent = self
        e.parent_dist = distance
        self.update_hr(pmtree, e)

    def dmin(self, p, pmtree):
        d1 = pmtree.distance_fn(p, self.pivot, **pmtree.distance_params) - self.radius
        d2 = float("-inf")
        for t in range(pmtree.nhr):
            d2 = max(d2, max(pmtree._query_pivot_distance[t] - self.hr[t][1],
                             self.hr[t][0] - pmtree._query_pivot_distance[t]))
        return max(0, d1, d2)


    def dmax(self, p, pmtree):
        node_to_query_distance = pmtree.distance_fn(p, self.pivot, **pmtree.distance_params)
        return min(node_to_query_distance + self.radius,
                   min([node_to_query_distance + self.hr[t][1] for t in range(pmtree.nhr)]))

class Entry(object):
    """Object representing an object in a PM-tree."""
    def __init__(self, e):
        self.entry = e#np.array(e) ni vec np.array ker lahko edit_distance nucas
        self.parent_dist = None
        self.pd = None
        self.parent = None

    def get_pivot(self):
        return self.entry

    def __eq__(self, other):
        if isinstance(self.entry, np.ndarray):
            return (self.entry == other.entry).all()
        else:
            return self.entry == other.entry

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        s = "point:%s\ndistance to parent:%f\nPD:%s\nparent:%s" % (str(self.entry), self.parent_dist,
                                                                   self.pd, str(self.parent))
        return s

_distances = {
    "minkowski": minkowski,
    "edit_distance": edit_distance
}


class PMTree(object):
    """Represents pivoted M-tree"""

    def __init__(self, data=None, m=100, p=0, nhr=None, npd=None, distance="minkowski", mink_p=2,
                 promote_fn="random", partition_fn="default", nr_pivot_groups=10):
        """
        Initializes a new instance of PM-tree.
        :param data: 2d array where each element is an array of real numbers representing one point in space
        :param m: maximum number of children in one node
        :param p: number of pivots
        :param nhr: number of ring regions for each node, requirement nhr <= p
        :param npd: each entry stores distances to first 'npd' pivots, requirement npd <= p
        :param distance: TODO: which distances are available!
        :param mink_p: integer defining which minkowski distance to use
        :param promote_fn: promote function used in split. Must accept list of nodes and return two pivot points
        (o1 and o2) of a node which will represent the pivots routing points of both nodes in split. It must not change
        the list of nodes. Two built-in methods are available, "random" and "mM_RAD", "random" is used by default.
        :param partition_fn: function which divides entries in two disjoint subsets. Parameters are (o1, o2, entries)
        where o1 and o2 are two pivot points of routing objects each defining a group and entries is a list of entries.
        It must return a tuple (group1, group2) where group1 is a list of entries that should have o1 as a pivot of
        rounding object and group2 are the other entries. By default generalized hyperplane method is used which is
        described in M-tree paper.
        """
        #TODO: imas za distance-e nastet katere lahko nuca.
        if data is None and p > 0 or data is not None and len(data) < p:
            raise ValueError("Number of pivots must not be greater than number of data points!")
        if nhr > p or npd > p:
            raise ValueError("Parameters nhr and npd must not be greater than number of pivots!")
        if m < 2:
            raise ValueError("Maximum number of children must be at least 2!")
        if nhr is None:
            nhr = p
        if npd is None:
            npd = p
        self.root = None
        self.size = 0
        self.height = 0
        self.p = p
        self.m = m
        self.nhr = nhr
        self.npd = npd
        self.promote_fn = promote_fn
        self.promote_fn_id = 0 #user defined function
        if isinstance(promote_fn, basestring):
            if promote_fn == "random":
                self.promote_fn = self._default_promote
                self.promote_fn_id = 1 #default random promote function
            elif promote_fn == "mM_RAD":
                self.promote_fn = self._mm_rad_promote
                self.promote_fn_id = 2 #mM_RAD promote function
            else:
                raise ValueError("The only built-in promote functions are \"random\" and \"mM_RAD\"!")
        elif not hasattr(self.promote_fn, '__call__'):
            raise ValueError("Parameter promote_fn must be a function or one of allowed strings!")
        self.partition_fn = partition_fn
        self.partition_fn_id = 0 #user defined function
        if isinstance(partition_fn, basestring):
            if partition_fn == "default":
                self.partition_fn = self._default_partition
                self.partition_fn_id = 1 #default partition function
            else:
                raise ValueError("The only built-in partition function is \"default\"!")
        elif not hasattr(self.partition_fn, '__call__'):
            raise ValueError("Parameter partition_fn must be a function or one of allowed strings!")
        self.distance_fn = None
        self.distance_type = None
        self.distance_params = {}
        if isinstance(distance, basestring):
            #set distance
            if not distance in _distances:
                raise ValueError("Invalid distance parameter!")
            self.distance_type = distance
            self.distance_fn = _distances[distance]
            if distance == "minkowski":
                self.distance_params["p"] = mink_p
        elif hasattr(distance, '__call__'):
            self.distance_fn = distance
        else:
            raise ValueError("Invalid distance argument!")
        #choose random p pivots
        best_pivots = None
        best_distance = float("-inf")
        for i in range(nr_pivot_groups):
            pivot_idxs = set([])
            while len(pivot_idxs) != p:
                pivot_idxs.add(random.randint(0, len(data)-1))
            cur_dist = sum(self.distance_fn(data[p1], data[p2], **self.distance_params)
                           for p1 in pivot_idxs for p2 in pivot_idxs if p1 != p2)
            if cur_dist > best_distance:
                best_distance = cur_dist
                best_pivots = pivot_idxs
        if p > 0:
            self.pivots = [data[i] for i in best_pivots]
        else:
            self.pivots = []
        if data is not None:
            for d in data:
                #create an entry and calculate distances from object to all pivots
                e = Entry(d)
                #insert in the tree
                self.insert(e)


    def _default_promote(self, nodes):
        """
        Default promote method, returns 2 random pivots.
        :param nodes: list of nodes
        :return (p1, p2) where p1 and p2 are two different pivots.
        """
        idxs = set([])
        while len(idxs) != 2:
            idxs.add(random.randint(0, len(nodes)-1))
        idxs = list(idxs)
        return nodes[idxs[0]].get_pivot(), nodes[idxs[1]].get_pivot()

    def _mm_rad_promote(self, nodes):
        """
        Confirmed mM_RAD promote, uses previous pivot as one of two new pivots.
        :param nodes: list of nodes
        :return: (p1, p2) where p1 and p2 are two different pivots.
        """
        p1 = nodes[0].parent.get_pivot() #first pivot is the pivot of the splitting node
        p2 = None
        min_r = None
        for node in nodes:
            cur_r = 0.0
            if isinstance(p1, np.ndarray) and all(node.get_pivot() == p1) or \
                    node.get_pivot() == p1:
                continue
            for node2 in nodes:
                if node == node2:
                    continue
                dmin = abs(node.parent_dist - node2.parent_dist)
                cur_r = max(cur_r, dmin)
            if min_r is None or cur_r < min_r:
                min_r = cur_r
                p2 = node.get_pivot()
        return p1, p2

    def _default_partition(self, o1, o2, nodes):
        """
        Partitions nodes in two groups.
        :param o1: pivot of first group
        :param o2: pivot of second group
        :param nodes: list of nodes
        :return: tuple (group1, group2) where group1 and group2 are non-empty disjoint subsets of nodes
        """
        group1, group2 = [], []
        for i, node in enumerate(nodes):
            d1 = self.distance_fn(o1, node.get_pivot(), **self.distance_params)
            d2 = self.distance_fn(o2, node.get_pivot(), **self.distance_params)
            if d1 < d2: #closer to o1 so it goes in group1
                group1.append(node)
            else:
                group2.append(node)
        if len(group1) == 0 or len(group2) == 0:
            print "ena groupa je prazna!"
            print "group1:", group1
            print "group2:", group2
            exit(0)
        return group1, group2

    ########DEBUG#########
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
        if node.is_leaf:
            s = "krog:("+str(node.pivot)+", "+ str(node.radius) + "), entries:"+str([x.entry for x in node.children])
            print '\t' * level + s
        else:
            print '\t' * level + str(node)+", krog:("+str(node.pivot)+", " + str(node.radius) + ")"
            for child in node.children:
                self.print_tree(child, level+1)

    def check_radius(self, node=None):
        if node is None:
            node = self.root
            if node is None:
                return True
        if isinstance(node, Entry):
            return True
        for child in node.children:
            dist_to_parent = self.distance_fn(child.get_pivot(), node.get_pivot(), **self.distance_params)
            if dist_to_parent > node.radius:
                print "napaka v radiju!"
                print "node:", node
                print "child:", child
                print "child pivot:", child.get_pivot()
                print "dejanski distance:", self.distance_fn(child.get_pivot(), node.get_pivot(), **self.distance_params)
                print "radij node-a:", node.radius
                self.print_tree()
                return False
            self.check_radius(child)

        return True

    def check_nr_children(self, node=None):
        """
        Checks if all nodes contain between <= m children.
        :param node: starting node, default = root
        :return: True if all nodes containg between <= m otherwise False
        """
        if node is None:
            node = self.root
            if node is None:
                return True
        if isinstance(node, Entry):
            return True
        if node == self.root:
            if not node.is_leaf and len(node.children) < 2:
                print "root je inner in nima vsaj dveh sinov"
                return False
        if len(node.children) > self.m:
            print "self.m:", self.m
            print "node nima pravilnega stevila sinov: node:%s, sons:%s" % (node, len(node.children))
            return False
        for x in node.children:
            if not self.check_nr_children(x):
                return False
        return True

    def check_parents(self, node=None):
        """
        Checks if all nodes have correct parents.
        :param node: subtree to check
        :return: True if correct otherwise False
        """
        if node is None:
            node = self.root
            if node is None:
                return True
            if node.parent is not None:
                print "root ima starsa wtf"
                return False
        if node.is_leaf:
            for child in node.children:
                if not isinstance(child, Entry) or child.parent != node or \
                                child.parent_dist != self.distance_fn(node.get_pivot(), child.get_pivot(),
                                                                      **self.distance_params):
                    print "child od leafa ni entry!!"
                    print "child:", child
                    print "type child:", type(child)
                    print "child.parent:", child.parent
                    print "child.parent_dist:", child.parent_dist
                    print "dejanski parent dist:", self.distance_fn(node.get_pivot(), child.get_pivot(),
                                                                      **self.distance_params)
                    print "dejanski parent:", node
                    print "type(node):", type(node)
                    return False
            return True
        if isinstance(node, Entry):
            return True
        #je inner node
        for child in node.children:
            if isinstance(child, Entry) or child.parent != node:
                print "napaka v parentih"
                print "parent:", node
                print "type(parent):", type(node)
                print "type(child):", type(child)
                print "child:", child
                print "parent od childa:", child.parent
                return False
            if not self.check_parents(child):
                return False
        return True

    def cover(self, node, e):
        """
        Returns true if all HR's of node cover entry
        :param node: node
        :param e: entry
        :return: true if all HR's cover entry e otherwise false
        """
        for i in range(self.nhr):
            if i < self.npd:
                d = e.pd[i]
            else:
                d = self.distance_fn(e.entry, self.pivots[i], **self.distance_params)
            if not node.hr[i][0] <= d <= node.hr[i][1]:
                return False
        return True

    def check_hrs(self, node=None):
        """
        Checks if all rings cover their points.
        :param node: starting node, default = root
        :return: True if all rings cover their points otherwise False
        """
        if node is None:
            node = self.root
            if node is None:
                return True, []
        if isinstance(node, Entry):
            return True, [node]
        entries = []
        for x in node.children:
            res, entries_below = self.check_hrs(x)

            if not res:
                print "FAIL HRS!"
            for entry in entries_below:
                if not self.cover(node, entry):
                    print "FAIL HRS"
                    print "node:", node
                    print "hrs od nodea:", node.hr
                    print "entry:", entry.entry
                    return False, []
            entries = entries + entries_below

        return True, entries

    def leaves_lvls(self, node=None, lvl=None):
        """
        Gets heights on which leaves are located in a tree.
        :param node: current node
        :param lvl: current height
        :return: set of heights of leaves
        """
        if node is None:
            node = self.root
        if lvl is None:
            lvl = 0 #root lvl
        if isinstance(node, Entry):
            return set([])
        res = set([])
        if node.is_leaf:
            res.add(lvl)
        for n in node.children:
            res = res.union(self.leaves_lvls(n, lvl+1))
        return res

    def get_entries(self,node):
        """
        Gets entries in node subtree.
        :param node: subtree
        :return: all entries which are in given subtree
        """
        if node is None:
            return []
        if node.is_leaf:
            return [x for x in node.children]
        else:
            res = []
            for child in node.children:
                res += self.get_entries(child)
            return res
    #######END DEBUG######
    #@profile
    def _choose_leaf(self, node, e, distance=None):
        """
        Inserts entry in a leaf node.
        :param node: current node while traversing
        :param e: entry
        :param distance: distance from entry to last visited node
        """
        if node.is_leaf:
            if len(node.children) == self.m:
                self._split(node, e)
            else:
                node.add_entry(self, e, distance)
        else:
            #update rings with entry
            node.update_hr(self, e)
            min_radius_enlargement = None
            min_pivot_dist = None
            best_child = None

            for child in node.children:
                pivot_dist = self.distance_fn(e.entry, child.pivot, **self.distance_params) #distance(e, child_pivot)
                enlargement = max(0.0, pivot_dist - child.radius) #radius enlargement needed
                if best_child is None or enlargement < min_radius_enlargement:
                    min_radius_enlargement = enlargement
                    min_pivot_dist = pivot_dist
                    best_child = child
                elif enlargement == min_radius_enlargement and pivot_dist < min_pivot_dist:
                    min_pivot_dist = pivot_dist
                    best_child = child
            #update radius of child if needed
            if min_pivot_dist > best_child.radius:
                best_child.radius = min_pivot_dist
            self._choose_leaf(best_child, e, min_pivot_dist)


    def _split(self, node, e):
        """
        Performs a split on a node and additional entry. Also adjusts the tree and performs additional splits if
        necessary.
        :param node: node to be split
        :param e: node/entry which couldn't be inserted because node was already full
        """
        node.add_entry(self, e, self.distance_fn(e.get_pivot(), node.get_pivot(), **self.distance_params))
        q = [x for x in node.children]
        node_parent = node.parent
        o1, o2 = self.promote_fn(q)
        o1_parent_distance = None
        o2_parent_distance = None
        if node != self.root:
            o1_parent_distance = self.distance_fn(o1, node_parent.pivot, **self.distance_params)
            o2_parent_distance = self.distance_fn(o2, node_parent.pivot, **self.distance_params)
        #creating new node
        new_node = Node(o2, 0.0, o2_parent_distance, [], node.is_leaf, nhr=self.nhr) #hr is set to infinity
        #fixing the old node
        node.pivot = o1
        node.parent_dist = o1_parent_distance
        node.radius = 0.0
        node.children = []
        node.hr = [[float('inf'), float('-inf')] for i in range(self.nhr)] #infinity
        #partition
        group1, group2 = self.partition_fn(o1, o2, q)
        node.add_children(self, group1)
        new_node.add_children(self, group2)
        if node == self.root:
            self.root = Node(o1, 0.0, None, children=[], is_leaf=False, hr=None, nhr=self.nhr)
            self.root.add_children(self, [node, new_node])
            self.height += 1
        else:
            if len(node_parent.children) == self.m:
                #is full, split it
                self._split(node_parent, new_node)
            else:
                node_parent.add_children(self, [new_node])


    def insert(self, e):
        """
        Inserts entry e in a tree.
        :param e: entry to insert
        """
        if not isinstance(e, Entry):
            raise ValueError("Parameter e must be an instance of class Entry.")
        #calculate distances to pivots
        e.pd = [self.distance_fn(e.entry, self.pivots[i], **self.distance_params) for i in range(self.npd)]
        #check if the tree is empty
        if self.root is None:
            #add e to it
            self.root = Node(e.entry, #point
                             0.0, #radius
                             None, #parent of root is None
                             [], #children
                             is_leaf=True, #root is leaf
                             hr=None,
                             nhr=self.nhr
                             #hr defines rings, must compute distances because nhr might be greater than npd
                            )
            self.root.add_entry(self, e, 0.0)
            self.height = 1
        else:
            #roots radius is never updated so update it before choosing leaf, HR is.
            d = self.distance_fn(self.root.pivot, e.entry, **self.distance_params)
            if self.root.radius < d:
                self.root.radius = d
            self._choose_leaf(self.root, e, distance=d) #also inserts, updates hr's and radiuses and performs splits

        self.size += 1


    """ --------------  SEARCH  ----------------"""

    def query(self, p, k=1):
        """
        Finds k-nearest neighbors of a point.
        :param p: point whose neighbors are needed
        :param k: number of nearest neighbors
        :return: tuple(neighbors, distances) where neighbors and distances are lists: i-th index in neighbor represents
        (i+1)-nearest neighbor while the same index in distance represents its distance to query point.
        """
        if k > self.size:
            raise ValueError("Cannot give you %d NNs from %d points!" % (k, self.size))
        if not self.root:
            raise ValueError("Tree is empty!")
        self.checked_entries = 0
        self.dist_count = 0
        self._kth_nearest = float("inf")
        self.nn_heap = [(float("-inf"), False) for i in range(k)] #heap containing n-nearest neighbors...
                                                                    # nn_heap = [(negative distance, entry)]
        self._query_pivot_distance = [self.distance_fn(p, self.pivots[i], **self.distance_params)
                                        for i in range(max(self.nhr, self.npd))]
        self.dist_count += max(self.nhr, self.npd) #pivot distances
        #self.nn_search(self.root, p, k)
        self._nn_search_iterative(p, k)
        neighbors = []
        distances = []
        while self.nn_heap:
            neighbor = heappop(self.nn_heap)
            neighbors.insert(0, neighbor[1].entry)
            distances.insert(0, abs(neighbor[0]))
        self._query_pivot_distance = None
        return neighbors, distances

    #@profile
    def nn_search(self, node, p, k):
        """
        Updates heap with k-nn of entry p.
        :param node: current node
        :param p: query point
        :param k: number of nearest neighbors needed
        """
        node_query_dist = self.distance_fn(node.pivot, p, **self.distance_params)
        if node.is_leaf:
            for e in node.children:
                kth_nearest = self.nn_heap[0][0]*-1.0
                skip = False
                for t in range(self.npd):
                    if not abs(self._query_pivot_distance[t] - e.pd[t]) <= kth_nearest:
                        skip = True
                        break
                if skip or not abs(node_query_dist - e.parent_dist) <= kth_nearest:
                    continue
                self.checked_entries += 1
                d = self.distance_fn(e.entry, p, **self.distance_params)
                if d < kth_nearest:
                    heappop(self.nn_heap) #odstrani prejsnjega
                    heappush(self.nn_heap, (-d, e)) #das novega noter
        else:

            #generate Active Branch List sorted by dmin
            a = sorted([(c, c.dmin(p, self)) for c in node.children], key= lambda x: x[1])
            for i in range(len(a)):
                kth_nearest = self.nn_heap[0][0]*-1.0
                if a[i][1] > kth_nearest:
                    return
                skip = False
                for t in range(self.nhr):
                    if not(self._query_pivot_distance[t] - kth_nearest <= a[i][0].hr[t][1] and \
                        self._query_pivot_distance[t] + kth_nearest >= a[i][0].hr[t][0]):
                        skip=True
                        break
                if skip or not abs(node_query_dist - a[i][0].parent_dist) <= kth_nearest + a[i][0].radius:
                    continue
                self.nn_search(a[i][0], p, k)


    def need_to_check(self, node):
        if isinstance(node, Entry):
            #check hyperrings
            for t in range(self.npd):
                if not abs(self._query_pivot_distance[t] - node.pd[t]) <= self._kth_nearest:
                    return False
            #check hypersphere
            if self.query_info[1] is not None and not abs(self.query_info[1] - node.parent_dist) <= self._kth_nearest:
                return False
            #get distance from query to this point
            self.checked_entries += 1
            self.dist_count += 1
            self.query_info[1] = self.distance_fn(self.query_info[0], node.get_pivot(), **self.distance_params)
            return self.query_info[1] <= self._kth_nearest

        else:
            #check hyperrings
            for t in range(self.nhr):
                if not(self._query_pivot_distance[t] - self._kth_nearest <= node.hr[t][1] and \
                        self._query_pivot_distance[t] + self._kth_nearest >= node.hr[t][0]):
                    return False
            #check hypersphere
            if self.query_info[1] is not None and not abs(self.query_info[1] - node.parent_dist) <= \
                                                            self._kth_nearest + node.radius:
                return False
            #get distance from query to this point
            self.query_info[1] = self.distance_fn(self.query_info[0], node.get_pivot(), **self.distance_params)
            self.dist_count += 1

            return self.query_info[1] <= self._kth_nearest + node.radius


    def _nn_search_iterative(self, p, k):
        self.query_info = [p, None] #[query_point, cur_dist]
        q = [] #heap of candidates
        cur_node = self.root
        while cur_node:
            saved_dist = self.query_info[1]
            for child in cur_node.children:
                self.query_info[1] = saved_dist #reset dist for every child
                if self.need_to_check(child):
                    #we must search it
                    if isinstance(child, Entry):
                        if self._kth_nearest > self.query_info[1]:
                            heappop(self.nn_heap)
                            heappush(self.nn_heap, (-self.query_info[1], child))
                            self._kth_nearest = self.nn_heap[0][0]*-1.0
                    else:
                        #insert child in priority queue
                        d2 = float("-inf")
                        for t in range(self.nhr):
                                d2 = max(d2, max(self._query_pivot_distance[t] - child.hr[t][1],
                                                 child.hr[t][0] - self._query_pivot_distance[t]))
                        dmin = max(0.0, self.query_info[1] - child.radius, d2)
                        heappush(q, (dmin, child, self.query_info[1]))

            if q:
                #choose next entry to check
                first = heappop(q)
                if first[0] >= self._kth_nearest:
                    #end the search
                    self.query_info = None
                    return
                else:
                    cur_node = first[1]
                    self.query_info[1] = first[2]
            else:
                return


    #for debugging
    def get_entries(self,node):
        """
        Gets entries in node subtree.
        :param node: subtree
        :return: all entries which are in given subtree
        """
        if node is None:
            return []
        if node.is_leaf:
            return [x for x in node.children]
        else:
            res = []
            for child in node.children:
                res += self.get_entries(child)
            return res

    def nn_search_brute(self,entries,p,k):
        """
        Performs brute-force k-nn search.
        :param entries: all entries
        :param p: query point
        :param k: number of nearest neighbors
        :param distance_p: defining which Lp-norm to use
        :return: list of tuples (distance, neighbor) sorted ascending
        """
        #entries = self.get_entries(self.root)
        #print "vseh vnosov brurte_force dobil:", len(entries)

        entries2 = [(self.distance_fn(x.entry, p, **self.distance_params), x) for x in entries]
        heapify(entries2)

        res = []
        while len(res) != k:
            res.append(heappop(entries2))
        return res

    """ --------------  SAVE/LOAD --------------"""
    def save(self, filename):
        """
        Saves PM-Tree in a specified file.

        :param filename: file in which PM-Tree is saved. It should have ".cpickle" extension.
        """
        #need to remove functions from attributes to pickle
        self.distance_fn = None
        self.promote_fn = None
        self.partition_fn = None
        f = open(filename,"wb")
        cPickle.dump(self, f)
        f.close()

    def load(self, filename, distance=None, promote_fn=None, partition_fn=None):
        """
        Loads PM-Tree from file.

        :param filename: file from which PM-Tree is loaded.
        :param distance: distance function, not needed if predefined function was used
        :return: PMTree instance
        """

        f = open(filename, "rb")
        self = cPickle.load(f)
        f.close()
        #setting right distance function
        if self.distance_type is None:
            if distance is None:
                raise ValueError("Distance function is needed!")
            self.distance_fn = distance
        else:
            self.distance_fn = _distances[self.distance_type]
        #setting right promote function
        if self.promote_fn_id == 1:
            self.promote_fn = self._default_promote
        elif self.promote_fn_id == 2:
            self.promote_fn = self._mm_rad_promote
        elif self.promote_fn_id == 0 and promote_fn is not None:
            self.promote_fn = promote_fn
        else:
            raise ValueError("Promote function is needed!")
        #setting right partition function
        if self.partition_fn_id == 1:
            self.partition_fn = self._default_partition
        elif self.partition_fn_id == 0 and partition_fn is not None:
            self.partition_fn = partition_fn
        else:
            raise ValueError("Partition function is needed!")

        return self
