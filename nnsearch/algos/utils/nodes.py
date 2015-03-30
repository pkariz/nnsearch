class Leaf(object):
	"""Leaf class represents a leaf node in R-tree."""
	def __init__(self, rect, entries):
		self.children = entries #entries = [Entry1,Entry2...]
		self.mbr = rect

class Inner(object):
	"""Inner class represents an inner node in R-tree"""
	def __init__(self, rect, children_rects):
		self.mbr = rect
		self.children = children_rects