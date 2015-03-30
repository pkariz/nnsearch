from nnsearch.datasets import load_dataset
from nnsearch.exact import RTree
from nnsearch.algos.utils import Entry

rtree_params = {
    "dimensions": None,
    "min_node_size": 2,
    "max_node_size": 5,
    "method": "linear"
}


class TestRTree:

    def test_insert(self):
        ds_name = "german_post_codes"
        ds = load_dataset(ds_name)
        tree = RTree()
        rtree_params["dimensions"] = len(ds.data[0])
        tree.build(**rtree_params)
        inserted = {}
        for i in range(len(ds.data)):
            e = Entry(ds.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                tree.insert(e)
                assert tree.index.check_mbrs() == True
                assert tree.index.check_parents() == True
                leaves = tree.index.leaves_lvls(tree.index.root)
                assert len(leaves) == 1
                assert list(leaves)[0] == tree.height-1
                assert tree.index.check_nr_rects() == True

        assert len(inserted.keys()) == tree.size == tree.index.size == len(tree.index.get_entries(tree.index.root))



    def test_delete(self):
        ds_name = "german_post_codes"
        ds = load_dataset(ds_name)
        tree = RTree()
        inserted = {}
        d = len(ds.data[0])
        rtree_params["dimensions"] = d
        tree.build(**rtree_params)
        for i in range(len(ds.data)):
            e = Entry(ds.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                tree.insert(e)

        entries = tree.index.get_entries(tree.index.root)
        deleted = 0
        for entry in entries:
            tree.delete(entry)
            deleted += 1
            assert tree.index.check_mbrs() == True
            parents = tree.index.check_parents()
            rects = tree.index.check_nr_rects()
            if not parents or not rects:
                print "FAIL: parents:%s, rects:%s" % (parents, rects)
                print "tree after delete:"
                tree.index.print_tree()
            assert rects == True
            assert parents == True
            leaves = tree.index.leaves_lvls(tree.index.root)
            assert tree.size == tree.index.size == len(entries)-deleted
            if tree.size != 0:
                assert len(leaves) == 1
                assert list(leaves)[0] == tree.height-1

        assert tree.index.root is None


    def test_query(self):
        ds_name = "german_post_codes"
        ds = load_dataset(ds_name)
        tree = RTree()
        inserted = {}
        d = len(ds.data[0])
        rtree_params["dimensions"] = d
        tree.build(**rtree_params)
        for i in range(1000):
            e = Entry(ds.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                tree.insert(e)

        entries = tree.index.get_entries(tree.index.root)
        k = 10
        for dd in [2]:#range(1,6):
            print "-------------------------------------------------------------"
            print "nov d:", dd
            print "-------------------------------------------------------------"
            distance_p = dd

            for i in range(1000, len(ds.data), 10):
                query_point = ds.data[i]
                nearest = tree.index.nn_search_brute(entries,query_point, k, distance_p)
                nearest_moj, distances = tree.query(query_point, k, mink_p=distance_p)
                assert set([x[0] for x in nearest]) == set(distances)
                assert len(nearest_moj) == len(distances) == len(nearest) == k


    def test_save_load(self):
        ds_name = "german_post_codes"
        ds = load_dataset(ds_name)
        tree = RTree()
        inserted = {}
        d = len(ds.data[0])
        rtree_params["dimensions"] = d
        tree.build(**rtree_params)
        for i in range(len(ds.data)):
            e = Entry(ds.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                tree.insert(e)

        tree_file = "rtree.data"
        tree.save(tree_file)
        loaded_tree = RTree()
        loaded_tree.load(tree_file)
        assert loaded_tree.size == tree.size
        assert loaded_tree.height == tree.height
        assert loaded_tree.algorithm == tree.algorithm == "R-Tree"
        entries = [tuple(x.entry.tolist()) for x in tree.index.get_entries(tree.index.root)]
        loaded_entries = [tuple(x.entry.tolist()) for x in loaded_tree.index.get_entries(loaded_tree.index.root)]
        assert len(entries) == len(loaded_entries)
        assert set(entries) == set(loaded_entries)
        assert loaded_tree.index.check_mbrs() == True
        assert loaded_tree.index.check_parents() == True
        leaves = loaded_tree.index.leaves_lvls(loaded_tree.index.root)
        assert len(leaves) == 1
        assert list(leaves)[0] == loaded_tree.height-1
        assert tree.index.d == loaded_tree.index.d
