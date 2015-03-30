import numpy as np
import time
from nnsearch.datasets import Dataset, load_dataset
from nnsearch.exact import PMTree
from nnsearch.algos.pmtree import Entry

pmtree_params = {
    "max_node_size": 5,
    "p": 10,
    "nhr": 10,
    "npd": 10,
    "distance": "minkowski",
    "mink_p": 2
}

mtree_params = {
    "max_node_size" : 5,
    "p" : 0,
    "nhr": 0,
    "npd": 0,
    "distance": "minkowski",
    "mink_p": 2
}

class TestPMTree:

    def test_insert(self):
        ds_name = "german_post_codes"
        ds = load_dataset(ds_name)
        tree = PMTree()
        tree.build(data=None, **mtree_params)
        inserted = {}
        entries = []
        it = 3000 #len(ds.data)
        for i in range(it):
            e = Entry(ds.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                entries.append(ds.data[i])
                tree.insert(e)

                assert tree.index.check_parents() == True
                assert tree.index.check_nr_children() == True
                assert tree.index.check_radius() == True
                assert tree.index.check_hrs()[0] == True
                leaves = tree.index.leaves_lvls(tree.index.root)
                assert len(leaves) == 1
                assert list(leaves)[0] == tree.height-1


    def test_query(self):
        ds_name = "german_post_codes"
        ds_prev = load_dataset(ds_name)
        tree = PMTree()
        it = 2000
        inserted = {}
        entries = []
        for i in range(it):
            e = Entry(ds_prev.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                entries.append(ds_prev.data[i])
        ds = Dataset(name="test", data=np.array(entries))
        tree.build(ds, **pmtree_params)

        entries = tree.index.get_entries(tree.index.root)
        k = 30
        for dd in [2]:#range(1,6):
            checked_entries = []
            # print "-------------------------------------------------------------"
            # print "nov d:", dd
            # print "-------------------------------------------------------------"
            jjj = 0
            times_moj = []
            times_bf = []
            for i in range(it, len(ds_prev.data), 100):
                query_point = ds_prev.data[i]
                start = time.time()
                nearest = tree.index.nn_search_brute(entries,query_point, k)
                elapsed = time.time() - start
                times_bf.append(elapsed)
                start = time.time()
                nearest_moj, distances = tree.query(query_point, k)
                elapsed = time.time() - start
                times_moj.append(elapsed)
                assert set([x[0] for x in nearest]) == set(distances)
                assert len(nearest_moj) == len(distances) == len(nearest) == k
                jjj += 1
                checked_entries.append(tree.index.checked_entries)


    def test_save_load(self):
        ds_name = "german_post_codes"
        ds_prev = load_dataset(ds_name)
        tree = PMTree()
        it = 300
        inserted = {}
        entries = []
        for i in range(it):
            e = Entry(ds_prev.data[i])
            if not inserted.has_key((e.entry[0], e.entry[1])):
                inserted[(e.entry[0], e.entry[1])] = 1
                entries.append(ds_prev.data[i])
        ds = Dataset(name="test", data=np.array(entries))
        tree.build(ds, **pmtree_params)
        tree_file = "pmtree.data"
        tree.save(tree_file)
        loaded_tree = PMTree()
        loaded_tree.load(tree_file)
        assert True == True
        assert loaded_tree.size == tree.size
        assert loaded_tree.height == tree.height
        assert loaded_tree.algorithm == tree.algorithm == "PM-Tree"
        assert loaded_tree.index.check_parents() == True
        assert loaded_tree.index.check_nr_children() == True
        assert loaded_tree.index.check_radius() == True
        assert loaded_tree.index.check_hrs()[0] == True
        entries = [tuple(x.entry.tolist()) for x in tree.index.get_entries(tree.index.root)]
        loaded_entries = [tuple(x.entry.tolist()) for x in loaded_tree.index.get_entries(loaded_tree.index.root)]
        assert len(entries) == len(loaded_entries)
        assert set(entries) == set(loaded_entries)
        leaves = loaded_tree.index.leaves_lvls(loaded_tree.index.root)
        assert len(leaves) == 1
        assert list(leaves)[0] == loaded_tree.height-1
        loaded_tree.insert(Entry(ds_prev.data[it+1]))
        assert loaded_tree.size == tree.size+1

