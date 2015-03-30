import numpy as np
from heapq import heappop, heappush, heapify #storing k-nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.nodes import Inner, Leaf
from utils.rectangle import Rect
from utils.entry import Entry
import cPickle


def mink2(x, y, p):
    """
    Returns minkowski distance between points x and y.
    :param x: first vector
    :param y: second vector
    :param p: defining Lp-norm
    :return: distance between x and y using Lp-norm
    """
    res = 0.0
    for i in range(len(x)):
        res += (max(x[i], y[i]) - min(x[i], y[i]))**p
    return res


class RSTree(object):
    """
    R*-Tree implementation.
    """

    def __init__(self, dimensions, min_nr_rects=2, max_nr_rects=5, p=0.3):
        """
        Initializes a new instance of RS-tree.
        :param dimensions: number of dimensions
        :param min_nr_rects: minimum number of rectangles in a node
        :param max_nr_rects: maximum number of rectangles in a node
        :param p: number between 0 and 1 representing the ratio of reinserted childrens of a node
        """
        self.min_nr_rects = min_nr_rects
        self.max_nr_rects = max_nr_rects
        self.reinsert_p = p
        self.root = None
        self.size = 0
        self.height = 0
        self.d = dimensions
        if min_nr_rects >= max_nr_rects:
            raise ValueError("Invalid number of rectangles. Maximum number of rectangles must be higher than minumum number.")

    def clear(self):
        """
        Clears R*-tree making it empty.
        """
        self.root = None
        self.height = 0
        self.size = 0

    def update_mbr(self, node):
        """
        Updates minimum bounding rectangle of a node.
        """
        node.mbr = Rect.mbr([child.mbr for child in node.children])

    def print_tree(self, node=None, level=0):
        """
        Prints tree in an awkward way which is good enough for debugging.
        :param node: node whose subtree is printed
        :param level: integer to know the current height from the starting node
        """
        if node:
            node = self.root
            print "-----------printing tree------------"
            if self.root is None:
                print "Tree is empty!"
                return
        s = ""
        if isinstance(node, Leaf):
            s = str(node)+": mbr:"+str(node.mbr)+", entries:"+str([x.entry if isinstance(x, Entry) else x for x in node.children])
            print '\t' * level + s
        else:
            print '\t' * level + str(node)+", mbr:", str(node.mbr)
            for child in node.children:
                self.print_tree(child,level+1)

    def add_children(self,node,children):
        """
        Adds children from list 'children' to node 'node'.
        :param node: node to which children are added
        :param children: iterable containing children to add
        """

        for child in children:
            node.children.append(child)
            child.parent = node
            node.mbr = Rect.mbr([node.mbr,child.mbr])

    #@profile
    def choose_subtree(self, n, e, lvl):
        """
        Chooses leaf in which entry should be inserted.
        :param n: current node while traversing
        :param e: entry
        :param lvl: height on which entry should be inserted
        :return: leaf in which entry should be inserted
        """
        if lvl == 0:
            return n
        if isinstance(n, Leaf):
            print "WTF sem v leaf in ni lvl, lvl:", lvl
            print "correct mbrs:", self.check_mbrs()
            print "correct parents:", self.check_parents()
            print "parent of n:", type(n.parent)
            print "fail leaves, ", self.leaves_lvls(self.root)
            exit(0)
        best_child = None
        if isinstance(n.children[0], Leaf):
            #children are leaves
            best_overlap_enlargement = None
            best_volume_enlargement = None
            for i in range(len(n.children)):
                #least overlap enlargement
                overlap_enlargement = Rect.get_overlap_enlargement(n, i, e)
                if best_child is None or overlap_enlargement < best_overlap_enlargement:
                    best_child = n.children[i]
                    best_overlap_enlargement = overlap_enlargement
                    best_volume_enlargement = None
                elif overlap_enlargement == best_overlap_enlargement:
                    #least volume enlargement
                    if best_volume_enlargement is None:
                        best_volume_enlargement = best_child.mbr.volume_increase(e.mbr)
                    volume_enlargement = n.children[i].mbr.volume_increase(e.mbr)
                    if volume_enlargement < best_volume_enlargement:
                        best_child = n.children[i]
                        best_volume_enlargement = volume_enlargement
                    elif volume_enlargement == best_volume_enlargement:
                        #least volume
                        if n.children[i].mbr.volume < best_child.mbr.volume:
                            best_child = n.children[i]
        else:
            min_enlargement = None#float("inf")
            #choose 'best' child
            for child in n.children:
                new_enlargement = child.mbr.volume_increase(e.mbr)
                if  best_child is None or new_enlargement < min_enlargement:
                    min_enlargement = new_enlargement
                    best_child = child
                elif new_enlargement == min_enlargement:
                    if child.mbr.volume < best_child.mbr.volume:
                        best_child = child
        return self.choose_subtree(best_child, e, lvl-1)

    """ --------------  NODE SPLITTING --------------"""
    def choose_split_axis(self, node):
        """
        Chooses axis for split.
        :param node: node which is overflown
        :return: integer representing axis --> 0 is 1st dimension etc
        """
        best_margin = None
        best_axis = None
        for i in range(self.d):
            #sort entries by lower value in dimension i
            s1 = sorted(node.children, key= lambda x: x.mbr.s[i])
            #sort entries by higher value in dimension i
            s2 = sorted(node.children, key= lambda x: x.mbr.t[i])
            margin = 0.0
            for s in [s1,s2]:
                #generate distributions
                for j in range(self.max_nr_rects - 2*self.min_nr_rects+2):
                    first_group = s[:self.min_nr_rects+j]
                    second_group = s[self.min_nr_rects+j:]
                    margin += Rect.mbr([x.mbr for x in first_group]).margin() + Rect.mbr([x.mbr for x in second_group]).margin()
            if best_axis is None or margin < best_margin:
                best_axis = i
                best_margin = margin

        return best_axis


    def split(self, node):
        """
        Performs a split on a node.
        :param node: node which needs to be split
        """
        axis = self.choose_split_axis(node)
        #choose split index
        best_overlap = None
        best_volume = None
        best_distribution = None
        #sort entries by lower value in dimension i
        s1 = sorted(node.children, key= lambda x: x.mbr.s[axis])
        #sort entries by higher value in dimension i
        s2 = sorted(node.children, key= lambda x: x.mbr.t[axis])

        for s in [s1,s2]:
            overlap = 0.0
            volume = 0.0
            #generate distributions
            for j in range(self.max_nr_rects - 2*self.min_nr_rects+2):
                first_group = s[:self.min_nr_rects+j]
                second_group = s[self.min_nr_rects+j:]
                mbr_first_group = Rect.mbr([x.mbr for x in first_group])
                mbr_second_group = Rect.mbr([x.mbr for x in second_group])
                overlap += Rect.intersect(mbr_first_group, mbr_second_group)
                volume += mbr_first_group.volume + mbr_second_group.volume
            if best_distribution is None or overlap < best_overlap:
                best_distribution = (first_group, second_group)
                best_overlap = overlap
                best_volume = volume
            elif overlap == best_overlap:
                #minimum volume
                if volume < best_volume:
                    best_distribution = (first_group, second_group)
                    best_volume = volume

        #drugi nov node mora bit leaf ce je node leaf, drugace mora bit inner, skratka ista morata bit
        #create a new node which will contain first group of best distribution
        if isinstance(node, Leaf):
            b = Leaf(best_distribution[0][0].mbr, [best_distribution[0][0]])
            best_distribution[0][0].parent = b
            self.add_children(b, best_distribution[0][1:])
        else:
            b = Inner(best_distribution[0][0].mbr, [best_distribution[0][0]])
            best_distribution[0][0].parent = b
            self.add_children(b, best_distribution[0][1:])

        #change the current node to contain second group of best distribution
        node.children = [best_distribution[1][0]]
        node.mbr = best_distribution[1][0].mbr
        best_distribution[1][0].parent = node
        self.add_children(node, best_distribution[1][1:])

        return node, b


    def split_node(self, node):
        return self.split(node)


    """ --------------  INSERT NODE --------------"""
    #@profile
    def insert_data(self, e):
        """
        Inserts entry in a tree.
        :param e: entry
        """
        self.overflown_levels = {}
        if self.root != None:
            self.insert(e, self.height-1)
        else:
            self.insert(e, 0)
        self.size+=1

    #@profile
    def reinsert(self, node, lvl):
        """
        Reinserts some children of node on the same level.
        :param node: node
        :param lvl: level on which children should be inserted
        """

        node_center = [(node.mbr.t[i] - node.mbr.s[i])/2.0 for i in range(node.mbr.d)]
        s = sorted(node.children, key = lambda x:
                                            mink2(np.array(node_center), np.array([(x.mbr.t[i] - x.mbr.s[i])/2.0
                                                                                   for i in range(x.mbr.d)]),2),
                   reverse=True)
        nr_remove = int(len(s)*self.reinsert_p)
        if nr_remove == 0:
            nr_remove = 1
        node.children = s[nr_remove:]
        self.update_mbr(node)
        height_before = self.height
        for i in range(nr_remove):
            s[i].parent = None
            new_height = self.height #while reinserting the root may split so the height may increase
            self.insert(s[i], lvl + (new_height-height_before))


    def overflow_treatment(self, node, lvl):
        """
        Reinserts some children of node if node is not root and this is the first time calling this function with this
        level during insert. Otherwise performs a split on node and returns new nodes.
        :param node: node
        :param lvl: height
        :return: tuple (node, ll) where ll is the second node in case the split was performed otherwise ll is None
        """
        if node != self.root and lvl not in self.overflown_levels:
            #reinsert
            self.overflown_levels[lvl] = 1
            self.reinsert(node, lvl)
            return node, None
        else:
            #split
            return self.split(node)

    #@profile
    def insert(self, e, lvl):
        """
        Insert entry or node in a tree on specific level
        :param e: entry or node
        :param lvl: height on which e is inserted
        """
        #check if the tree is empty
        if self.root == None:
            #add e to it
            self.root = Leaf(e.mbr, [e])
            e.parent = self.root
            self.root.parent = None #root doesnt have a parent
            self.height = 1
        else:
            n = self.choose_subtree(self.root, e, lvl)
            try:
                self.add_children(n, [e])
            except:
                print "n:", str(n)
                print "type(n):", type(n)
                print "n.parent:", type(n.parent)
                print "correct mbrs:", self.check_mbrs()
                print "correct parents:", self.check_parents()
                print "fail leaves, ", self.leaves_lvls(self.root)
                print "lvl:", lvl
                print "lvl od tega n-ja:", self._get_level(n)
                print "st vseh:", self.size
                exit(0)
            ll = None
            while len(n.children) > self.max_nr_rects:
                #n is overflown, split or reinsert
                if n == self.root:
                    n, ll = self.overflow_treatment(n, lvl)
                    self.root = Inner(Rect.mbr([n.mbr, ll.mbr]), [n, ll])
                    self.root.parent = None
                    n.parent = self.root
                    ll.parent = self.root
                    self.height += 1
                    break
                else:
                    n, ll = self.overflow_treatment(n, lvl)
                    if ll is None:
                        #reinsert happened
                        break
                    else:
                        #split happened
                        p = n.parent
                        #self.update_mbr(nn) prej
                        p.children.append(ll)
                        ll.parent = p

                #go up the tree
                lvl -= 1
                n = n.parent

            #update mbrs on path up to the root from the last node (last split node or node where reinsert happened or node where data was put)
            while n != None:
                self.update_mbr(n)
                n = n.parent


    """ --------------  DELETE NODE --------------"""
    def update_mbr_all(self, node):
        """
        Updates MBR of all nodes.
        :param node: subtree to be updated
        """
        if isinstance(node, Entry):
            return
        for child in node.children:
            self.update_mbr_all(child)
        if len(node.children) > 0:
            self.update_mbr(node)

    def check_mbrs(self, node=None):
        """
        Checks if all MBR's are correct.
        :param node: subtree to check
        :return: True if correct otherwise False
        """
        if node is None:
            node = self.root
            if node is None:
                return True
        if isinstance(node, Entry):
            return True
        for child in node.children:
            self.check_mbrs(child)
        prev = node.mbr
        self.update_mbr(node)
        if node.mbr != prev:
            return False
        return True

    def check_nr_rects(self, node=None):
        """
        Checks if all nodes contain between m and M children.
        :param node: starting node, default = root
        :return: True if all nodes containg between m and M children otherwise False
        """
        if node is None:
            node = self.root
            if node is None:
                return True
        if isinstance(node, Entry):
            return True
        if node == self.root:
            if isinstance(node, Inner) and len(node.children) < 2:
                print "root je inner in nima vsaj dveh sinov"
                return False
        if not self.min_nr_rects <= len(node.children) <= self.max_nr_rects and node != self.root:
            print "node nima pravilnega stevila sinov: node:%s, sons:%s" % (node, len(node.children))
            return False
        for x in node.children:
            if not self.check_nr_rects(x):
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
        if isinstance(node, Leaf):
            for child in node.children:
                if not isinstance(child, Entry) or child.parent != node:
                    """print "child od leafa ni entry!!"
                    print "child:", child
                    print "type child:", type(child)
                    print "child.parent:", child.parent
                    print "dejanski parent:", node
                    print "type(node):", type(node)"""
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

    #@profile
    def delete(self, e):
        """
        Deletes entry from the tree
        :param e: entry
        :return: True if entry was deleted otherwise False (if tree doesnt contain that entry)
        """
        if self.root is None:
            return False
        l = self.find_leaf(self.root, e)
        if not l:
            return False
        l.children.remove(e)
        e.parent = None
        self.size -= 1
        self.condense_tree(l, set([]), self.height-1)

        #shorten tree if it contains only 1 entry
        if isinstance(self.root, Leaf) and len(self.root.children) == 0 \
            or isinstance(self.root, Inner) and len(self.root.children) == 0:
            #restart root, because its empty
            self.root = None
            self.height = 0
        elif isinstance(self.root, Inner) and len(self.root.children) == 1:
            #there is only 1 so set it to be the root
            self.root = self.root.children[0]
            self.root.parent = None
            self.height -= 1
        else:
            self.update_mbr(self.root) # to je v bistvu n, ostale sem ali popravu alpa odstranu

        return True



    def _get_level(self, node, cur_node = None, lvl=0):
        """
        Returns height on which node is located. Root has height 0.
        :param cur_node: current node while traversing
        :param node: node whose height is needed
        :param lvl: current height
        :return: height of node
        """
        if cur_node is None:
            cur_node = self.root

        if type(cur_node) == type(node) and cur_node == node:
            return lvl

        for child in cur_node.children:
            #check if there is overlap between mbr of entry 'e' and child
            if Rect.overlap(child.mbr,node.mbr):
                res = self._get_level(node, child, lvl+1)
                if res:
                    return res
        return None

    #@profile
    def condense_tree(self, n, qs, lvl):
        #print "condense, trenuten node:%s, lvl:%d" % (str(n), lvl)
        """
        Updates tree recursively fixing nodes which underflow because delete operation was performed. Reinserts some
        entries and inner nodes.
        :param n: current node
        :param qs: nodes to be reinserted
        """
        if n == self.root and len(qs) > 0:
            for q, h, height_before in qs: #each 'e' should be inserted on height 'h', same as before
                for e in q.children:
                    if isinstance(q, Leaf):
                        self.size -= 1 #cuz insert is gonna do size++
                        self.insert_data(e)
                    else:
                        #height might change while reinserting so need to insert it on the right height
                        """parents_before = self.check_parents()
                        print "TREE BEFORE ANOTHER REINSERT: reinserting:%s on height:%d, sprememba height:%d" % \
                              (str(e), h+(self.height-height_before), (self.height-height_before))
                        self.print_tree()"""
                        self.insert(e, h+(self.height-height_before))
                        if not self.check_parents():
                            print "FAIL PARENTS!!!!"
                            print "rects:", self.check_nr_rects()
                            self.print_tree()
                            exit(0)

        elif n != self.root:
            pn = n.parent
            if len(n.children) < self.min_nr_rects:
                pn.children.remove(n)
                n.parent = None #dodano
                qs.add((n, lvl, self.height))
            else:
                self.update_mbr(n)
            self.condense_tree(pn, qs, lvl-1)


    def leaves_lvls(self, node, lvl=None):
        """
        Gets heights on which leaves are located in a tree.
        :param node: current node
        :param lvl: current height
        :return: set of heights of leaves
        """
        if node is None:
            return set([])
        if lvl is None:
            lvl = 0 #root lvl
        if isinstance(node, Entry):
            return set([])
        res = set([])
        if isinstance(node, Leaf):
            res.add(lvl)
        for n in node.children:
            res = res.union(self.leaves_lvls(n, lvl+1))
        return res

    def find_leaf(self, node, e):
        """
        Returns a leaf which contains the given entry. If there is no such leaf it returns None.
        :param e: instance of class Entry
        :return: leaf which contains entry
        """

        if isinstance(node, Leaf):
            for entry in node.children:
                if entry == e:
                    return node
            return None
        else:
            for child in node.children:
                #check if there is overlap between mbr of entry 'e' and child
                if Rect.overlap(child.mbr,e.mbr):
                    res = self.find_leaf(child,e)
                    if res:
                        return res
            return None

    """ --------------  NN-SEARCH --------------"""

    def query(self, p, k=1, distance_p = 2):
        """
        Finds k-nearest neighbors of a point using Lp-norm distance.
        :param p: point whose neighbors are needed
        :param k: number of nearest neighbors
        :param distance_p: integer representing which Lp-norm to use
        :return: tuple(neighbors, distances) where neighbors and distances are lists: i-th index in neighbor represents
        (i+1)-nearest neighbor while the same index in distance represents its distance to query point.
        """
        if k > self.size:
            raise ValueError("Cannot give you %d NNs from %d points!" % (k, self.size))
        if not self.root:
            raise ValueError("Tree is empty!")
        self.checked_entries = 0
        self.dist_count = 0
        self.nn_heap = [] #heap containing n-nearest neighbors...nn_heap = [(negative distance, entry)]

        #self.nn_search(self.root, p, k, distance_p)
        self._nn_search_iterative(p, k, distance_p)

        neighbors = []
        distances = []
        while self.nn_heap:
            neighbor = heappop(self.nn_heap)
            neighbors.insert(0, neighbor[1].entry)
            distances.insert(0, abs(neighbor[0])**(1/float(distance_p)))
        return neighbors, distances


    def nn_search(self, node, p, k, distance_p):
        """
        Updates heap with k-nn of entry p using Lp-norm.
        :param node: current node
        :param p: query point
        :param k: number of nearest neighbors needed
        :param distance_p: integer representing which Lp-norm to use
        """
        if isinstance(node, Leaf):
            #poglej razdaljo vsakega entry-ja
            for e in node.children:
                self.checked_entries += 1
                d = mink2(e.entry, p, distance_p)
                if len(self.nn_heap) < k:
                    heappush(self.nn_heap, (-d, e)) #negativen distance v heap ker O(1) za najmanjsega
                else:
                    worst = self.nn_heap[0]
                    if d < worst[0] * -1.0:
                        heappop(self.nn_heap) #odstrani prejsnjega
                        heappush(self.nn_heap, (-d, e)) #das novega noter
        else:
            #generate Active Branch List sorted by minDist ascending
            a = sorted(node.children, key = lambda x: x.mbr.minDist(p, distance_p))
            for i in range(len(a)):
                if len(self.nn_heap) < k:
                    self.nn_search(a[i], p, k, distance_p)
                else:
                    worst = self.nn_heap[0]
                    if a[i].mbr.minDist(p, distance_p) > worst[0] * -1.0:
                        #min razdalja od node-a do tocke p je vecja kot najvecja razdalja iz heap-a
                        return
                    else:
                        self.nn_search(a[i], p, k, distance_p)


    def _nn_search_iterative(self, p, k, distance_p):
        q = [] #heap of candidates
        cur_node = self.root
        self.nn_heap = [(float("-inf"), False) for i in range(k)]
        self._kth_nearest = self.nn_heap[0][0] * -1.0
        while cur_node:
            if isinstance(cur_node, Leaf):
                for child in cur_node.children:
                    child_dist = mink2(child.entry, p, distance_p)
                    self.checked_entries += 1
                    self.dist_count += 1
                    if self._kth_nearest > child_dist:
                        heappop(self.nn_heap)
                        heappush(self.nn_heap, (-child_dist, child))
                        self._kth_nearest = self.nn_heap[0][0]*-1.0
            else:
                for child in cur_node.children:
                    mindist = child.mbr.minDist(p, distance_p)
                    self.dist_count += 1
                    if mindist > self._kth_nearest:
                        continue#break
                    #insert child in priority queue
                    heappush(q, (mindist, child))

            if q:
                #choose next node to check
                first = heappop(q)
                if first[0] >= self._kth_nearest:
                    #end the search
                    return
                else:
                    cur_node = first[1]
            else:
                return

    """ --------------  NN-SEARCH BRUTE-FOCE --------------"""
    def get_entries(self,node):
        """
        Gets entries in node subtree.
        :param node: subtree
        :return: all entries which are in given subtree
        """
        if node is None:
            return []
        if isinstance(node, Leaf):
            return [x for x in node.children]
        else:
            res = []
            for child in node.children:
                res += self.get_entries(child)
            return res
    def nn_search_brute(self,entries,p,k,distance_p):
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

        entries2 = [(mink2(x.entry, p, distance_p), x) for x in entries]
        heapify(entries2)

        res = []
        while len(res) != k:
            res.append(heappop(entries2))
        return res

    """ --------------  PLOT TREE --------------"""
    def plot_tree(self, filename=None, marker_size = 2, height=None):
        """
        Plots tree using matplotlib. Only able to plot 2d and 3d trees.
        :param filename: file in which the image is stored
        :param marker_size: marker_size to use in plot
        :param height: on which height rectangles should be plotted, default is -1 = all rectangles
        """
        if self.d != 2 and self.d != 3:
            print "Can only plot 2d and 3d tree."
            return
        if not self.root:
            print "Tree is empty."
            return
        if height is None:
            height = self.height
        self.colors = ['#3366ff','r','g','b','m','y']
        self.linewidths = [2.0/float(i+1) for i in range(0, self.height-1)]
        y_range = self.root.mbr.t[1]-self.root.mbr.s[1]
        x_range = self.root.mbr.t[0]-self.root.mbr.s[0]
        z_range = -1
        if self.d == 3:
            z_range = self.root.mbr.t[2]-self.root.mbr.s[2]
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            plt.clf() #clear plot
        self.plot2(self.root, marker_size, height)
        #nastavi range osi
        if self.d == 3:
            ax.set_zbound(self.root.mbr.s[2]-z_range/10.0, self.root.mbr.t[2]+z_range/10.0) # range for z-axis
            ax.set_ylim3d(self.root.mbr.s[1]-y_range/10.0,self.root.mbr.t[1]+y_range/10.0) # range for y-axis
            ax.set_xlim3d(self.root.mbr.s[0]-x_range/10.0, self.root.mbr.t[0]+x_range/10.0) # range for x-axis
        else:
            plt.axis([self.root.mbr.s[0]-x_range/10.0, self.root.mbr.t[0]+x_range/10.0,
                      self.root.mbr.s[1]-y_range/10.0, self.root.mbr.t[1]+y_range/10.0]) #range y in x na sliki
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()

    def plot3d_rect(self, rect, level):
        """
        Plots 3d rectangle, level defines color and linewidth.
        """
        #spodnji pravokotnik
        t = rect.t
        s = rect.s
        color = self.colors[level%len(self.colors)]
        linewidth = self.linewidths[level]
        plt.plot([s[0],t[0]],[s[1],s[1]],[s[2],s[2]],linewidth = linewidth, color=color)
        plt.plot([s[0],t[0]],[t[1],t[1]],[s[2],s[2]],linewidth = linewidth, color=color)
        plt.plot([s[0],s[0]],[s[1],t[1]],[s[2],s[2]],linewidth = linewidth, color=color)
        plt.plot([t[0],t[0]],[s[1],t[1]],[s[2],s[2]],linewidth = linewidth, color=color)

        #zgornji pravokotnik
        plt.plot([s[0],t[0]],[s[1],s[1]],[t[2],t[2]],linewidth = linewidth, color=color)
        plt.plot([s[0],t[0]],[t[1],t[1]],[t[2],t[2]],linewidth = linewidth, color=color)
        plt.plot([s[0],s[0]],[s[1],t[1]],[t[2],t[2]],linewidth = linewidth, color=color)
        plt.plot([t[0],t[0]],[s[1],t[1]],[t[2],t[2]],linewidth = linewidth, color=color)

        #vmesne
        plt.plot([s[0],s[0]],[s[1],s[1]],[s[2],t[2]],linewidth = linewidth, color=color)
        plt.plot([t[0],t[0]],[s[1],s[1]],[s[2],t[2]],linewidth = linewidth, color=color)
        plt.plot([t[0],t[0]],[t[1],t[1]],[s[2],t[2]],linewidth = linewidth, color=color)
        plt.plot([s[0],s[0]],[t[1],t[1]],[s[2],t[2]],linewidth = linewidth, color=color)

    def plot2(self, node, marker_size, height, level=0):
        """
        Recursively plots the tree.
        """
        if isinstance(node, Leaf):
            xs, ys, zs = [],[],[]
            for e in node.children:
                xs.append(e.entry[0])
                ys.append(e.entry[1])
                if self.d == 3:
                    zs.append(e.entry[2])
            if self.d == 3:
                plt.plot(xs, ys, zs, 'k.',markersize=marker_size)
            else:
                plt.plot(xs, ys, 'k.',markersize=marker_size)
        else:
            for child in node.children:
                if self.d == 3 and level < height-1:
                    self.plot3d_rect(child.mbr, level)
                elif level < height-1:
                    w = child.mbr.t[0]-child.mbr.s[0]
                    h = child.mbr.t[1]-child.mbr.s[1]
                    p = mpatches.Rectangle(child.mbr.s, w, h, facecolor="none", linewidth = self.linewidths[level], edgecolor=self.colors[level%len(self.colors)])
                    p.set_linestyle("solid")
                    plt.gca().add_patch(p)
                self.plot2(child, marker_size, height, level+1)

    """ --------------  SAVE/LOAD --------------"""
    def save(self, filename):
        """
        Saves RS-Tree in a specified file.

        :param filename: file in which RS-Tree is saved. It should have ".cpickle" extension.
        """
        f = open(filename,"wb")
        cPickle.dump(self,f)
        f.close()

    def load(self, filename):
        """
        Loads RS-Tree from file.

        :param filename: file from which RS-Tree is loaded.
        :return: RSTree instance
        """
        f = open(filename,"rb")
        self = cPickle.load(f)
        f.close()
        return self

