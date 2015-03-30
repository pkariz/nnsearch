from heapq import heappop, heappush, heapify #storing k-nn
import random #random testi
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


class RTree(object):
    """
    R-Tree implementation with linear and quadratic split methods.
    """
    methods = ["linear","quadratic"]

    def __init__(self, dimensions, min_nr_rects=2, max_nr_rects=5, method="linear"): #could have leafsize as parameter
        """
        Initializes a new instance of R-tree.
        :param dimensions: number of dimensions
        :param min_nr_rects: minimum number of rectangles in a node
        :param max_nr_rects: maximum number of rectangles in a node
        :param method: split method either "linear" or "quadratic"
        """
        self.min_nr_rects = min_nr_rects
        self.max_nr_rects = max_nr_rects
        self.root = None
        self.size = 0
        self.height = 0
        self.d = dimensions
        if min_nr_rects >= max_nr_rects:
            raise ValueError("Invalid number of rectangles. Maximum number of rectangles must be higher than minumum number.")
        if method not in self.methods:
            raise ValueError("Invalid split method. Supported split methods are:",self.methods)
        self.method = method

    def clear(self):
        """
        Clears R-tree making it empty.
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
        if node is None:
            node = self.root
            print "-----------printing tree------------"
            if self.root is None:
                print "Tree is empty!"
                return
        s = ""
        if isinstance(node, Leaf):
            #s = "Leaf:"+str(node)+" --> "+str([x.entry for x in node.children])#"mbr:"+str(node.mbr)+", entries:"+str([x.entry for x in node.children])
            s = "Leaf:"+str(node)+" --> "+"mbr:"+str(node.mbr)+", entries:"+str([x.entry for x in node.children])
            print '\t' * level + s
        else:
            print '\t' * level + str(node)+", Inner"," mbr:", str(node.mbr)
            for child in node.children:
                self.print_tree(child,level+1)

    def add_children(self, node, children):
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
    def choose_leaf(self, n, e):
        """
        Chooses leaf in which entry should be inserted.
        :param n: current node while traversing
        :param e: entry
        :return: leaf in which entry should be inserted
        """
        if isinstance(n, Leaf):
            return n
        best_child = None
        min_enlargement = None#float("inf")
        for child in n.children:
            new_enlargement = child.mbr.volume_increase(e.mbr)
            if  best_child is None or new_enlargement < min_enlargement:
                min_enlargement = new_enlargement
                best_child = child
            elif new_enlargement == min_enlargement:
                if child.mbr.volume < best_child.mbr.volume:
                    best_child = child
        return self.choose_leaf(best_child, e)

    """ --------------  NODE SPLITTING --------------"""
    def linear_pick_seeds(self, q):
        """
        Returns two indexes representing starting seeds
        :param q: node which is overflown
        :return: tuple (i,j) which represents indexes of starting seeds; i < j
        """
        best_high, best_low = None, None
        cur_high, cur_low = None, None
        best_separation = None
        for d in range(self.d):
            lowest_high_side = float('inf') #najmanjsa zgornja tocka
            highest_low_side = float('-inf') #najvecja spodnja tocka
            left, right = float('inf'), float('-inf') #najbolj levo in desno v trenutni dimenziji
            for i in range(len(q)):
                if q[i].mbr.s[d] < left:
                    left = q[i].mbr.s[d]
                if q[i].mbr.t[d] > right:
                    right = q[i].mbr.t[d]
                if q[i].mbr.t[d] < lowest_high_side:
                    lowest_high_side = q[i].mbr.t[d]
                    cur_low = i
                if q[i].mbr.s[d] > highest_low_side:
                    highest_low_side = q[i].mbr.s[d]
                    cur_high = i
            if cur_high == cur_low or right == left:
                continue
            #normaliziras (delis z razponom na teji dimenziji)
            separation = abs(highest_low_side-lowest_high_side) / (right-left)
            if not best_separation or separation > best_separation:
                best_high = cur_high
                best_low = cur_low

        if not best_separation:
            best_high = 0
            best_low = 1
        return min(best_high,best_low), max(best_high,best_low)

    def linear_pick_next(self,q):
        """
        Returns index of a random element in q.
        :param q: list of entries which were not put in a node yet
        :return: random element from a node
        """
        return random.randint(0,len(q)-1)

    #TODO: area ali volume cim manjsi?
    def quadratic_pick_seeds(self,q):
        """
        Returns indexes of the entries that are most distant from each other.
        :param q: node which is overflown
        :return: tuple (i,j) which represents indexes of starting seeds; i < j
        """
        res = None
        best_delta = None
        for i in range(len(q)-1):
            for j in range(i+1,len(q)):
                rect1 = q[i].mbr #mbr of this object
                rect2 = q[j].mbr
                cur_delta = Rect.new_mbr_volume(rect1,rect2) - rect1.volume - rect2.volume
                if not res or cur_delta > best_delta:
                    res = (i,j)
                    best_delta = cur_delta
        return res

    #TODO: area ali volume difference?
    def quadratic_pick_next(self, q, a, b):
        """
        Returns index of the next entry to be considered.
        :param q: list of entries which were not put in a node yet
        :param a: node A
        :param b: node B
        """
        res = None
        max_diff = None
        for i in range(len(q)):
            e_mbr = q[i].mbr
            d1 = Rect.new_mbr_volume(a.mbr,e_mbr) - a.mbr.volume
            d2 = Rect.new_mbr_volume(b.mbr,e_mbr) - b.mbr.volume
            if not res or abs(d1-d2) > max_diff:
                res = i
                max_diff = abs(d1-d2)
        return res

    #TODO: area ali volume?
    #TODO2: sej nemore splitat node, ki ni leaf?

    def split(self, node):
        """
        Performs a split on a node.
        :param node: node which needs to be split
        :return: two nodes which are the result of a split of a given node
        """

        q = [x for x in node.children] #copy children
        if self.method == "linear":
            i,j = self.linear_pick_seeds(q)
        else:
            #quadratic
            i, j = self.quadratic_pick_seeds(q)
        #first pop object on position j because j > i
        r = q.pop(j)
        l = q.pop(i)

        node.children = [l]
        node.mbr = l.mbr
        l.parent = node
        #drugi nov node mora bit leaf ce je node leaf, drugace mora bit inner, skratka ista morata bit
        if isinstance(node, Leaf):
            b = Leaf(r.mbr,[r])
            r.parent = b
        else:
            b = Inner(r.mbr,[r])
            r.parent = b

        best = None
        while q:
            #check if all remaining children need to be put in the smallest group to satisfy minimum number of rectangles in a node property
            if len(node.children) + len(q) == self.min_nr_rects:
                self.add_children(node,q)
                return node,b
            elif len(b.children) + len(q) == self.min_nr_rects:
                self.add_children(b,q)
                return node,b
            else:
                if self.method == "linear":
                    i = self.linear_pick_next(q)
                else:
                    #quadratic
                    i = self.quadratic_pick_next(q, node, b)
                o = q.pop(i)
                node_inc = node.mbr.volume_increase(o.mbr)
                b_inc = b.mbr.volume_increase(o.mbr)
                if node_inc < b_inc:
                    #add to node
                    best = node
                elif node_inc > b_inc:
                    #add to b
                    best = b
                #elif node.mbr.volume() < b.mbr.volume():
                elif node.mbr.volume < b.mbr.volume:
                    #add to node
                    best = node
                elif node.mbr.volume > b.mbr.volume:
                    #add to b
                    best = b
                elif len(node.children) < len(b.children):
                    #add to node
                    best = node
                else:
                    #add to b, even if they have the same number of children
                    best = b
                if best == node:
                    #dodaj v a
                    self.add_children(node,[o])
                else:
                    #dodaj v b
                    self.add_children(b,[o])
        return node, b


    def split_node(self, node):
        return self.split(node)

    #TODO: returna ne tistega bool
    #@profile
    def adjust_tree(self, l, ll):
        """
        Adjusts MBR's and performs additional splits if necessary.
        :param l: the node which had to be split
        :param ll: the second node in split process, null if there was no split
        """
        n = l
        nn = ll
        while n != self.root:
            p = n.parent
            self.update_mbr(n)
            n = p #set n to next node, going up
            if nn:
                p.children.append(nn)
                nn.parent = p
                p.mbr = Rect.mbr([p.mbr, nn.mbr]) #novo
                nn = None #reset ll
                if len(p.children) > self.max_nr_rects:
                    #split needed
                    n, nn = self.split_node(p)
        self.update_mbr(n) #prej
        if nn:
            self.root = Inner(Rect.mbr([n.mbr,nn.mbr]),[n,nn])
            self.root.parent = None
            n.parent = self.root
            nn.parent = self.root
            self.height += 1

    """ --------------  INSERT NODE --------------"""
    #@profile
    def insert_data(self, e):
        """
        Insert entry in a tree
        :param e: entry
        """
        if not isinstance(e, Entry):
            raise ValueError("Parameter e must be an instance of class Entry.")
        #check if the tree is empty
        if self.root == None:
            #add e to it
            self.root = Leaf(e.mbr, [e])
            e.parent = self.root
            self.root.parent = None #root doesnt have a parent
            self.height = 1
        else:
            leaf=None
            leaf = self.choose_leaf(self.root, e)
            self.add_children(leaf, [e])
            ll = None
            if len(leaf.children) > self.max_nr_rects:
                #overflow, split node
                leaf, ll = self.split_node(leaf)
            self.adjust_tree(leaf, ll)
        self.size += 1

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




    def _insert_node(self, cur_node, node, lvl):
        """
        Inserts node in the tree on height 'lvl' to keep leaves on the same height.
        :param cur_node: current node while searching for the place to insert the node
        :param node: node to insert
        :param lvl: height on which node should be inserted
        """

        if lvl == 0 or isinstance(cur_node.children[0], Leaf):
            #insert it in this node
            cur_node.children.append(node)
            node.parent = cur_node
            cur_node.mbr = Rect.mbr([cur_node.mbr,node.mbr])
            ll = None
            if len(cur_node.children) > self.max_nr_rects:
                #overflow, split node
                cur_node, ll = self.split_node(cur_node)

            self.adjust_tree(cur_node, ll)

        else:
            best_child = None
            min_enlargement = float("inf")
            #choose 'best' child
            for child in cur_node.children:
                new_enlargement = child.mbr.volume_increase(node.mbr)
                if  new_enlargement < min_enlargement:
                    min_enlargement = new_enlargement
                    best_child = child
                elif new_enlargement == min_enlargement:
                    if child.mbr.volume < best_child.mbr.volume:
                        best_child = child
            self._insert_node(best_child,node, lvl-1)

    def _get_level(self,cur_node, node, lvl=0):
        """
        Returns height on which node is located. Root has height 0.
        :param cur_node: current node while traversing
        :param node: node whose height is needed
        :param lvl: current height
        :return: height of node
        """
        if type(cur_node) == type(node) and cur_node == node:
            return lvl
        if isinstance(cur_node, Leaf):
            return None
        else:
            for child in cur_node.children:
                #check if there is overlap between mbr of entry 'e' and child
                if Rect.overlap(child.mbr,node.mbr):
                    res = self._get_level(child,node, lvl+1)
                    if res:
                        return res
            return None


    def condense_tree(self, n, qs, lvl):
        #print "condense, trenuten node:%s, lvl:%d" % (str(n), lvl)
        """
        Updates tree recursively fixing nodes which underflow because delete operation was performed. Reinserts some entries
        and inner nodes.
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
                        self._insert_node(self.root, e, h+(self.height-height_before))
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
    #@profile
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
                    heappush(self.nn_heap,(-d,e)) #negativen distance v heap ker O(1) za najmanjsega
                else:
                    worst = self.nn_heap[0]
                    if d < worst[0] * -1.0:
                        heappop(self.nn_heap) #odstrani prejsnjega
                        heappush(self.nn_heap,(-d,e)) #das novega noter
        else:
            #generate Active Branch List sorted by minDist ascending
            a = sorted([(c.mbr.minDist(p, distance_p), c) for c in node.children], key=lambda x: x[0])
            for i in range(len(a)):
                if len(self.nn_heap) < k:
                    self.nn_search(a[i][1], p, k, distance_p)
                else:
                    worst = self.nn_heap[0]
                    if a[i][0] > worst[0] * -1.0:
                        #min razdalja od node-a do tocke p je vecja kot najvecja razdalja iz heap-a
                        return
                    else:
                        self.nn_search(a[i][1], p, k, distance_p)


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
                        continue
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
    def plot_tree(self, filename=None, marker_size=2, height=None):
        """
        Plots tree using matplotlib. Only able to plot 2d and 3d trees.
        :param filename: file in which the image is stored
        :param marker_size: marker_size to use in plot
        :param height: on which height rectangles should be plotted, default is None = all rectangles
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
            #plt.axis('equal')
            plt.axis([self.root.mbr.s[0]-x_range/10.0,self.root.mbr.t[0]+x_range/10.0,self.root.mbr.s[1]-y_range/10.0,self.root.mbr.t[1]+y_range/10.0]) #range y in x na sliki
        #plt.draw()
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
        Saves R-Tree in a specified file.

        :param filename: file in which R-Tree is saved. It should have ".cpickle" extension.
        """
        f = open(filename,"wb")
        cPickle.dump(self,f)
        f.close()

    def load(self, filename):
        """
        Loads R-Tree from file.

        :param filename: file from which R-Tree is loaded.
        :return: RTree instance
        """
        f = open(filename,"rb")
        self = cPickle.load(f)
        f.close()
        return self