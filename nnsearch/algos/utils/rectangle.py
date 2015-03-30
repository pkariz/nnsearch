import numpy as np
import sys
import math
from operator import mul


class Rect(object):

    def __init__(self, s, t):
        """Constructor for Rectangle. It takes two points which determine rectangle, For all k=1 to d => t_k > s_k.

        :param s: d dimensional point, represented with tuple (s1,s2...sd), numpy array or list
        :param t: d dimensional point, represented with tuple (t1,t2...td), numpy array or list
        """
        self.s = map(float,s)
        self.t = map(float,t)
        self.d = len(s)
        self.volume = self.volume()

    def __str__(self):
        res = ""
        for i in range(self.d):
            res += "(%f,%f)" % (self.s[i], self.t[i])
        return res

    def __eq__(self, other):
        if self.d != other.d or self.volume != other.volume:
            return False
        ss = self.s == other.s
        tt = self.t == other.t
        return ss and tt

    def __ne__(self, other):
        return not self == other

    def minDist(self, q, p):
        """Returns minimum distance from point to rectangle self, needs sqrt to get the actual distance.
        :param q: point
        :param p: integer to know which Lp norm to use
        """
        res = 0.0
        ri = -1
        for i in range(0,len(q)):
            if q[i] < self.s[i]:
                ri = self.s[i]
            elif q[i] > self.t[i]:
                ri = self.t[i]
            else:
                ri = q[i]
            res += abs(math.pow((q[i] - ri), p))#abs(q[i] - ri)**p
        #return math.sqrt(res)
        return res #ne korenim!


    def volume(self):
        """Returns the volume of a rectangle."""
        #return np.prod([self.t[i] - self.s[i] for i in range(self.d)])
        return reduce(mul, [self.t[i] - self.s[i] for i in range(self.d)], 1) #faster

    #TODO: ali dela tudi v 2d? neki mi vracalo 2x vecjo menda enkrat
    def area(self):
        """Returns the area of a rectangle."""
        res = 0.0
        for i in range(self.d):
            face_area = 1
            for j in range(self.d):
                if i != j:
                    face_area *= self.t[j] - self.s[j]
            res += face_area
        return 2 * res

    @staticmethod
    def mbr(rects):
        """Returns a new rectangle which is the MBR of all rectangles in list 'rects'."""
        if len(set(r.d for r in rects)) != 1:
            raise ValueError("Rectangles must have the same number of dimensions.")
        #find min and max in every dimension
        s_new, t_new = [],[]

        """for i in range(rects[0].d):
            s_new.append(reduce(lambda acc,b: min(b.s[i],acc),rects, sys.maxint))#min([r.s[i] for r in rects]))
            t_new.append(reduce(lambda acc,b: max(b.t[i],acc), rects, -sys.maxint-1))#max([r.t[i] for r in rects]))
        return Rect(s_new, t_new)"""
        for i in range(rects[0].d):
            cur_min = sys.maxint
            cur_max = -sys.maxint-1
            for r in rects:
                if r.s[i] < cur_min:
                    cur_min = r.s[i]
                if r.t[i] > cur_max:
                    cur_max = r.t[i]
            s_new.append(cur_min)
            t_new.append(cur_max)
        return Rect(s_new, t_new)


    @staticmethod
    #@profile
    def new_mbr_volume(a, b):
        """Returns a volume of MBR of rectangles 'a' and 'b'."""
        #find min and max in every dimension
        #return reduce(mul,[max(a.t[i], b.t[i]) - min(a.s[i], b.s[i]) for i in xrange(a.d)], 1.0)
        res = 1.0
        for i in range(a.d):
            res *= max(a.t[i], b.t[i]) - min(a.s[i], b.s[i])
        return res


    def volume_increase(self,r):
        """Returns the increase in volume of mbr if we add rectangle 'r' to the current rectangle"""
        #return Rect.mbr([self,r]).volume() - self.volume()
        #return Rect.new_mbr_volume([self,r]) - self.volume
        #return Rect.mbr_new_volume(self,r) - self.volume
        return Rect.new_mbr_volume(self,r) - self.volume


    def max_dim_increase(self,r):
        new = Rect.mbr([self,r])
        maxx = -1
        for i in range(new.d):
            prev = r.t[i] - r.s[i] #razpon prej po teji dimenziji
            now = new.t[i] - new.s[i] #nov razpon
            if now-prev > maxx:
                maxx =  abs(now-prev)
        return maxx


    @staticmethod
    def intersect(rect1, rect2):
        """
        Gets intersection volume between two rectangles.
        :param rect1: first rectangle
        :param rect2: second rectangle
        :return: intersection volume
        """
        res = 1.0
        for i in range(rect1.d):
            minmax = min(rect1.t[i], rect2.t[i])
            maxmin = max(rect1.s[i], rect2.s[i])
            if minmax <= maxmin:
                #they dont intersect in this dimension therefore there is no intersection
                return 0.0
            res *= minmax - maxmin
        return res

    @staticmethod
    #@profile
    def get_overlap_enlargement(node, e_idx, data):
        """
        Gets overlap of entry with other entries in node.children.
        overlap(e_idx) = sum{i=1, len(node.children),i != e_idx} ( volume(Ei.intersect(Ee_idx)) )
        :param node: node containing entries
        :param e_idx: idx of entry in node.children
        :param data: entry to add
        :return: overlap of entry with node.children (except with entry itself)
        """
        res = 0.0
        ek = node.children[e_idx].mbr
        ek_enlarged = Rect.mbr([ek, data.mbr])
        for j in range(len(node.children)):
            if j == e_idx:
                continue
            cur_rect = node.children[j].mbr
            res += Rect.intersect(ek_enlarged, cur_rect) - Rect.intersect(ek, cur_rect)
        return res

    def margin(self):
        """
        Gets margin of rectangle. Margin is sum of lengths in dimensions.
        :return: margin of rectangle
        """
        res = 0.0
        for i in range(self.d):
            res += self.t[i] - self.s[i]
        return res


    @staticmethod
    def overlap(rect1, rect2):
        """Returns True if rectangles rect1 and rect2 overlap. Otherwise returns False.
        Parameters:
            rect1 - instance of class Rect
            rect2 - instance of class Rect
        """
        for i in range(rect1.d):
            #ce se v kaksni dimenziji ne prekriva potem return false
            if rect1.s[i] > rect2.t[i] or rect2.s[i] > rect1.t[i]:
                return False
        return True
