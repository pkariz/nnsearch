import sys
from nnsearch.approx import Annoy, RKDTree, BoundaryF, FlannAuto, HKmeans, LSHFlann, LSHNearPy
from nnsearch.exact import BallTree, Brute, KDTree, KDTreeScikit, RTree, RSTree, PMTree
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance
from nnsearch.datasets import Dataset, load_dataset, samples
from nnsearch.flannindex import FlannIndex
import time
import datetime
import os
import random
import numpy as np
import cPickle
import math
import gc

#annoy parameters
annoy_params1 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 1
}

annoy_params10 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 10
}

annoy_params30 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 30
}

annoy_params60 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 60
}

annoy_params100 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 100
}

annoy_params200 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 200
}

annoy_params500 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : 500
}

annoy_params_neg1 = {
    "dimensions" : None,
    "distance" : "euclidean",
    "trees" : -1
}
#------------------------------------------------------

#ball-tree parameters
ball_tree_params20 = {
    "leaf_size" : 20,
    "distance" : "euclidean"
}

#flann autotune
flann_params9 = {
    "precision" : 0.9
}

flann_params7 = {
    "precision" : 0.7
}

flann_params99 = {
    "precision" : 0.99
}

#RKD-tree
rkdtree_params = {
    "trees" : 16,
    "precision": 0.99,
    "checks": 5000
}

rkdtree_params2 = {
    "trees" : 4,
    "precision": 0.7
}

#KD-tree scikit
kdtree_scikit_params = {
    "leaf_size" : 30,
    "metric" : "euclidean"
}

#kmeans
kmeans_params = {
    "branching" : 32,
    "iterations" : 5,
    "centers_init" : "random",
    "cb_index" : 0.5,
    "precision": 0.7,
}

kmeans_params2 = {
    "branching" : 32,
    "iterations" : 5,
    "centers_init" : "default",
    "cb_index" : 0.2,
    "precision": 0.99,
    "checks" : 15000
}

#lsh-flann
lshflann_params_6_12_0 = {
    "nr_tables" : 6,
    "key_size" : 12,
    "multi_probe_level" : 0 #standard lsh
}
lshflann_params_6_12_0_99 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 0, #standard lsh
    "precision" : 0.99
}

lshflann_params_6_12_0_99 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 0, #standard lsh
    "precision" : 0.99
}

lshflann_params_6_12_0 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 0
}

lshflann_params_6_12_1 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 1
}

lshflann_params_6_12_2 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 2
}

lshflann_params_6_12_2_99 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 2,
    "precision" : 0.99
}

lshflann_params_10_12_2 = {
    "nr_tables" : 10,
    "key_size" :12,
    "multi_probe_level" : 2
}

lshflann_params_10_12_0 = {
    "nr_tables" : 10,
    "key_size" :12,
    "multi_probe_level" : 0
}

lshflann_params_10_12_2_99 = {
    "nr_tables" : 10,
    "key_size" :12,
    "multi_probe_level" : 2,
    "precision" : 0.99
}

lshflann_params_10_12_0_99 = {
    "nr_tables" : 10,
    "key_size" :12,
    "multi_probe_level" : 0, #standard lsh
    "precision" : 0.99
}

lshflann_params_6_12_3 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 3
}
lshflann_params_6_12_5 = {
    "nr_tables" : 6,
    "key_size" :12,
    "multi_probe_level" : 5
}



#lsh-nearpy
lshnearpy_params5 = {
    "lshashes" : [RandomBinaryProjections('default', 5)],
    "distance" : EuclideanDistance()
}

lshnearpy_params10 = {
    "lshashes" : [RandomBinaryProjections('default', 10)],
    "distance" : EuclideanDistance()
}

lshnearpy_params20 = {
    "lshashes" : [RandomBinaryProjections('default', 20)],
    "distance" : EuclideanDistance()
}

lshnearpy_params50 = {
    "lshashes" : [RandomBinaryProjections('default', 50)],
    "distance" : EuclideanDistance()
}

lshnearpy_params100 = {
    "lshashes" : [RandomBinaryProjections('default', 100)],
    "distance" : EuclideanDistance()
}

#r-tree
rtree_params_l_5 = {
    "dimensions" : None,
    "max_node_size" : 5,
    "method" : "linear"
}

rtree_params_q_5 = {
    "dimensions" : None,
    "max_node_size" : 5,
    "method" : "quadratic"
}

rtree_params_l_25 = {
    "dimensions" : None,
    "max_node_size" : 25,
    "method" : "linear"
}

rtree_params_q_25 = {
    "dimensions" : None,
    "max_node_size" : 25,
    "method" : "quadratic"
}

rtree_params_l_100 = {
    "dimensions" : None,
    "max_node_size" : 100,
    "method" : "linear"
}

rtree_params_q_100 = {
    "dimensions" : None,
    "max_node_size" : 100,
    "method" : "quadratic"
}

#r*-tree
rstree_params_5 = {
    "dimensions" : None,
    "max_node_size" : 5
}

rstree_params_25 = {
    "dimensions" : None,
    "max_node_size" : 25
}

rstree_params_100 = {
    "dimensions" : None,
    "max_node_size" : 100
}

#pm-tree
mtree_params_5 = {
    "max_node_size" : 5,
    "p" : 0,
    "nhr": 0,
    "npd": 0,
    "distance": "minkowski",
    "mink_p": 2
}

mtree_params_25 = {
    "max_node_size" : 25,
    "p" : 0,
    "nhr": 0,
    "npd": 0,
    "distance": "minkowski",
    "mink_p": 2
}

mtree_params_100 = {
    "max_node_size" : 100,
    "p" : 0,
    "nhr": 0,
    "npd": 0,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_5_4_4 = {
    "max_node_size" : 5,
    "p" : 4,
    "nhr": 4,
    "npd": 4,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_25_4_4 = {
    "max_node_size" : 25,
    "p" : 4,
    "nhr": 4,
    "npd": 4,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_100_4_4 = {
    "max_node_size" : 100,
    "p" : 4,
    "nhr": 4,
    "npd": 4,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_5_32_4 = {
    "max_node_size" : 5,
    "p" :32,
    "nhr": 32,
    "npd": 4,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_25_32_4 = {
    "max_node_size" : 25,
    "p" :32,
    "nhr": 32,
    "npd": 4,
    "distance": "minkowski",
    "mink_p": 2
}


pmtree_params_100_32_4 = {
    "max_node_size" : 100,
    "p" :32,
    "nhr": 32,
    "npd": 4,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_5_64_8 = {
    "max_node_size" : 5,
    "p" : 64,
    "nhr": 64,
    "npd": 8,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_25_64_8 = {
    "max_node_size" : 25,
    "p" : 64,
    "nhr": 64,
    "npd": 8,
    "distance": "minkowski",
    "mink_p": 2
}

pmtree_params_100_64_8 = {
    "max_node_size" : 100,
    "p" : 64,
    "nhr": 64,
    "npd": 8,
    "distance": "minkowski",
    "mink_p": 2
}

#boundaryForest
bf_params_5_10_true = {
    "trees": 5,
    "max_node_size": 10,
    "parallel": True
}
bf_params_5_10_false = {
    "trees": 5,
    "max_node_size": 10,
    "parallel": False
}

bf_params_10_10_true = {
    "trees": 10,
    "max_node_size": 10,
    "parallel": True
}

bf_params_10_10_false = {
    "trees": 10,
    "max_node_size": 10,
    "parallel": False
}

bf_params_30_10_true = {
    "trees": 30,
    "max_node_size": 10,
    "parallel": True
}

bf_params_30_10_false = {
    "trees": 30,
    "max_node_size": 10,
    "parallel": False
}

bf_params_10_50_true = {
    "trees": 10,
    "max_node_size": 50,
    "parallel": True
}

bf_params_10_50_false = {
    "trees": 10,
    "max_node_size": 50,
    "parallel": False
}

bf_params_50_50_true = {
    "trees": 50,
    "max_node_size": 50,
    "parallel": True
}

bf_params_50_50_false = {
    "trees": 50,
    "max_node_size": 50,
    "parallel": False
}



algorithms_bruteforce = [
    ("Brute-force_cdef", Brute, {}, {}),
    #("Brute-force_c1", Brute, {"cores":1}, {"cores":1}),
    #("Brute-force_c2", Brute, {"cores":2}, {"cores":2}),
    #("Brute-force_c4", Brute, {"cores":4}, {"cores":4}),
]

algorithms_approx = [
    #("Annoy_1", Annoy, annoy_params1, {}),
    #("Annoy_10", Annoy, annoy_params10, {}),
    #("Annoy_30", Annoy, annoy_params30, {}),
    #("Annoy_60", Annoy, annoy_params60, {}),
    #("Annoy_100", Annoy, annoy_params100, {}),
    #("Annoy_200", Annoy, annoy_params200, {}),
    #("Annoy_500", Annoy, annoy_params500, {}),
    #("Annoy_-1", Annoy, annoy_params_neg1, {}),
    #("RKD-tree_cdef_7", RKDTree, rkdtree_params2, {}),
    #("RKD-tree_c1_99", RKDTree, rkdtree_params, {"cores":1}),
    ("RKD-tree_cdef_99", RKDTree, rkdtree_params, {}),  
    #("flann9", FlannAuto, flann_params9, {}),
    #("flann7", FlannAuto, flann_params7, {}),
    #("RKD-tree_c2", RKDTree, rkdtree_params, {"cores":2}),
    ("RKD-tree_c4", RKDTree, rkdtree_params, {"cores":4}),
    ("kmeans_99", HKmeans, kmeans_params, {}),
    #("lsh-nearpy_5", LSHNearPy, lshnearpy_params5, {}),
    #("lsh-nearpy_10", LSHNearPy, lshnearpy_params10, {}),
    #("lsh-nearpy_20", LSHNearPy, lshnearpy_params20, {}),
    #("BF_5_10_parallel", BoundaryF, bf_params_5_10_true, {}),
    #("BF_10_10_parallel", BoundaryF, bf_params_10_10_true, {}),
    #("BF_10_10_parallel_n10", BoundaryF, dict(bf_params_10_10_true.items() + {"n":10}.items()), {})
]

algorithms_exact = [
    ("Ball-tree_20", BallTree, ball_tree_params20, {}), #TODO: res exact?
    ("kd-tree_scikit", KDTreeScikit, kdtree_scikit_params, {}), #TODO: res exact?
    ("kd-tree_flann_cdef", KDTree, {}, {}),
    ("kd-tree_flann_c1", KDTree, {"cores":1}, {"cores":1}),
    ("kd-tree_flann_c2", KDTree, {"cores":2}, {"cores":2}),
    ("kd-tree_flann_c4", KDTree, {"cores":4}, {"cores":4}),
    #("R-tree_l_5", RTree, rtree_params_l_5, {}),
    #("R-tree_q_5", RTree, rtree_params_q_5, {}),
    #("R-tree_l_25", RTree, rtree_params_l_25, {}),
    #("R-tree_q_25", RTree, rtree_params_q_25, {}),
    #("R*-tree_5", RSTree, rstree_params_5, {}),
    #("R*-tree_25", RSTree, rstree_params_25, {}),
    #("M-tree_5", PMTree, mtree_params_5, {}),
    #("M-tree_25", PMTree, mtree_params_25, {}),
    ("PM-tree_5_4_4", PMTree, pmtree_params_5_4_4, {}),
    ("PM-tree_25_4_4", PMTree, pmtree_params_25_4_4, {}),
    #("PM-tree_5_32_4", PMTree, pmtree_params_5_32_4, {}),
    #("PM-tree_25_32_4", PMTree, pmtree_params_25_32_4, {}),
    #("PM-tree_5_64_8", PMTree, pmtree_params_5_64_8, {}),
    #("PM-tree_25_64_8", PMTree, pmtree_params_25_64_8, {})
]

timer = time.clock
timer_time = time.time
knns_neighbors = {}
knns_distances = {}
build_timeout = 1200 #in seconds
current_distance = "euclidean"

def get_precision(index, nearest, dists, query_idx, k):
    global knns_neighbors, knns_distances
    correct = 0
    eps = 0.000001
    for x in knns_distances[k][query_idx].tolist():
        if any(abs(x-y) < eps for y in dists.tolist()):
            correct += 1
    return correct / float(k)


def get_params(index, build_params, data):
    if isinstance(index, Annoy):
        return {"dimensions":len(data[0]), "metric": build_params["metric"]}
    elif isinstance(index, BoundaryF):
        res = {}
        if "d" in build_params:
            res["d"] = build_params["d"]
        if "dc" in build_params:
            res["dc"] = build_params["dc"]
        return res
    else: #isinstance(index, Brute) or isinstance(index, FlannIndex) or isinstance(index, LSHNearPy):
        return {}

def time_build(index, ds, params):
    start_cpu_time = time.time()
    index.build(data=ds, **params)
    build_cpu_time = time.time() - start_cpu_time
    return build_cpu_time, index

def get_build_info(algo_name, algo, params, ds, dataset_dir):
    global build_timeout
    index = algo()
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(os.path.join(dataset_dir, algo_name+".p")):
        sys.setrecursionlimit(500000000)
        try:
            build_cpu_time, index = timeout(time_build, args=(index, ds, params),
                                    timeout_duration=build_timeout, default=(None, None))
        except Exception, e:
            #memory allocation failed
            build_cpu_time = None
            print e
        if build_cpu_time is None:
            if index.algorithm == "BoundaryForest":
                #kill spawned processes
                try:
                    for process in index.index.procs:
                        if process.is_alive():
                            process.terminate()
                            print "process terminated"
                except Exception, e:
                    print e
            return None, None
        dont_pickle = ["BallTree-scikit", "Brute-force-flann", "KD-Tree-scikit", "BoundaryForest"]
        if index.algorithm not in dont_pickle and not isinstance(index, FlannIndex):#!= "BoundaryForest":
            try:
                index.save(os.path.join(dataset_dir, algo_name+".p"))
                with open(os.path.join(dataset_dir, algo_name+"_build_time.p"), "wb") as f:
                    cPickle.dump(build_cpu_time, f)
                params = get_params(index, params, ds.data)

                #pickle params
                with open(os.path.join(dataset_dir, algo_name+"_params.p"), "wb") as f:
                    cPickle.dump(params, f)
                index = algo()
                if isinstance(index, FlannIndex) or isinstance(index, BoundaryF):#flann and BF need data
                    index.load(os.path.join(dataset_dir, algo_name+".p"), ds)
                else:
                    index.load(os.path.join(dataset_dir, algo_name+".p"), **params)
            except Exception as e:
                print "couldnt pickle", algo_name
                print "reason:", e
                #pass
        else:
            print "%s, not saving it!" % (index.algorithm,)
    else:
        params = {}
        with open(os.path.join(dataset_dir, algo_name+"_params.p"), "rb") as f:
            params = cPickle.load(f)
        if isinstance(index, FlannIndex) or isinstance(index, BoundaryF):#flann and BF need data
            index.load(os.path.join(dataset_dir, algo_name+".p"), ds)
        else:
            try:
                index.load(os.path.join(dataset_dir, algo_name+".p"), **params)
            except Exception as e:
                print "failed to load index!"
                print "reason:", e
                exit(0)
        with open(os.path.join(dataset_dir, algo_name+"_build_time.p"), "rb") as f:
            build_cpu_time = cPickle.load(f)
    return build_cpu_time, index


def get_query_info(index, queries, ks, query_params):
    global knns
    global current_distance
    query_times = {k:[] for k in ks} #for timer
    query_times_time = {k:[] for k in ks} #for time.time
    query_times_all_clock = {} #all queries together with time.clock
    query_times_all_time = {} #all queries together with time.time
    precisions = {k:[] for k in ks} #check for exact
    missings = {k:[] for k in ks}
    checked = None
    dc = None
    has_checked = False
    if index.algorithm in ["PM-Tree", "R*-Tree", "R-Tree"]:
        has_checked = True
        checked = {k:[] for k in ks}
        dc = {k:[] for k in ks}
    #za flann da ni spikeov
    if isinstance(index, FlannIndex):
        for k in ks:
            for query_idx, query in enumerate(queries):
                _,_ = index.query(query, k, **query_params)
    for k in ks:
        for query_idx, query in enumerate(queries):
            gc.disable() #disable garbage collector
            start = timer()
            start2 = timer_time()
            nearest, dists = index.query(query, k, **query_params)
            elapsed = timer() - start
            elapsed2 = timer_time() - start2
            gc.enable() #enable it again
            query_times[k].append(elapsed)
            query_times_time[k].append(elapsed2)
            if index.algorithm == "Brute-force-flann":
                if k not in knns_neighbors:
                    knns_neighbors[k] = {}
                    knns_distances[k] = {}
                if current_distance == "euclidean":
                    dists = np.array([math.sqrt(y) for y in dists.tolist()])
                knns_distances[k][query_idx] = dists
                knns_neighbors[k][query_idx] = nearest
                precisions[k].append(1.0)
            else:
                if index.algorithm == "BoundaryForest":
                    #get actual distances from nearest to query point, BF returns distances of reduced vectors
                    dists = np.array([index.index.dist_x(query, y) for y in nearest])
                    """if k == 10 and query_idx == 0:
                        print "nearest 10 neighbors Forest:", nearest
                        print "nearest 10 dists:", dists"""
                elif index.algorithm == "AnnoyIndex":
                    dists = np.array([index.get_dist(query, index.index.get_item_vector(x)) for x in nearest.tolist()])
                elif isinstance(index, FlannIndex) and current_distance == "euclidean":
                    dists = np.array([math.sqrt(y) for y in dists.tolist()])
                precisions[k].append(get_precision(index, nearest, dists, query_idx, k))
            missings[k].append(k - len(dists))
            if has_checked:
                checked[k].append(index.index.checked_entries)
                dc[k].append(index.index.dist_count)
        #feed query with all queries at the same time instead 1 by 1
        if index.algorithm in ["PM-Tree", "R*-Tree", "R-Tree", "LSH-NearPy"]:
            query_times_all_clock[k] = -1.0
            query_times_all_time[k] = -1.0
            continue
        gc.disable() #disable garbage collector
        start3 = timer()
        start4 = timer_time()
        _, _ = index.query(queries, k, **query_params)
        elapsed3 = timer() - start3
        elapsed4 = timer_time() - start4
        gc.enable() #enable it again
        query_times_all_clock[k] = elapsed3
        query_times_all_time[k] = elapsed4

    return query_times, query_times_time, query_times_all_clock, query_times_all_time, precisions, missings, checked, dc

def save_info(algo_name, build_time, query_times, query_times_time, query_times_all_clock, query_times_all_time, precisions,
              missings, checked, dc, ks, dataset_dir):
    info = "build time:"+str(build_time)+"\n"
    qt = 3
    d = {"name": algo_name, "build_time": build_time, "query_times": query_times, "query_times_time": query_times_time,
         "query_times_all_clock": query_times_all_clock, "query_times_all_time": query_times_all_time,
         "precisions": precisions, "missings": missings, "checked": checked, "dc": dc}
    for k in ks:
        info += "----------NR OF QUERY POINTS:%d---------\n" % (k,)
        info += "---TIMER with time.clock()---:\n"
        info += "query_times[:3]:%s\n" % (str(query_times[k][:qt]),)
        info += "avg query time:%f\n" % (np.mean(query_times[k]),)
        info += "std of query times:%f\n" % (np.std(query_times[k]),)
        info += "query_times_all_clock:%f\n" % (query_times_all_clock[k],)
        info += "query_times_all_clock avg per query:%f\n" % (query_times_all_clock[k]/float(len(query_times[k])),)
        info += "---TIMER with time.time()---:\n"
        info += "query_times:%s\n" % (str(query_times_time[k][:qt]),)
        info += "avg query time:%f\n" % (np.mean(query_times_time[k]),)
        info += "std of query times:%f\n" % (np.std(query_times_time[k]),)
        info += "query_times_all_time:%f\n" % (query_times_all_time[k],)
        info += "query_times_all_time avg per query:%f\n" % (query_times_all_time[k]/float(len(query_times[k])),)
        info += "---STATS---:\n"
        info += "precision:%s\n" % (str(precisions[k][:qt]),)
        info += "avg precision:%f\n" % (np.mean(precisions[k]),)
        info += "std of precisions:%f\n" % (np.std(precisions[k]),)
        info += "missings:%s\n" % (str(missings[k][:qt]),)
        info += "avg missings:%f\n" % (np.mean(missings[k]),)
        info += "std of missings:%f\n" % (np.std(missings[k]),)
        info += "nr of non-empty missings:%d\n" % (len([x for x in missings[k] if x != 0][:qt]),)
        if checked is not None:
            info += "checked entries:%s\n" % (str(checked[k][:qt]),)
            info += "avg checked:%f\n" % (np.mean(checked[k]),)
            info += "std of checked:%f\n" % (np.std(checked[k]),)
            info += "distance computations:%s\n" % (str(dc[k][:qt]),)
            info += "avg dcs:%f\n" % (np.mean(dc[k]),)
            info += "std of dcs:%f\n" % (np.std(dc[k]),)
        info += "\n\n"
    filename = os.path.join(dataset_dir, algo_name+".txt")
    with open(filename, "w") as f:
        f.write(info)
    filename_pickle = os.path.join(dataset_dir, algo_name+"_info.p")
    with open(filename_pickle, "wb") as f:
        cPickle.dump(d, f)


def get_dir_name():
    """Returns the name of the directory."""
    now = datetime.datetime.today()
    rv = ""
    for x in ['year', 'month', 'day', 'hour', 'minute']:
        if x == 'hour':
            rv = rv[:-1] + "___"
        rv += str(getattr(now, x)) + "-"
    return rv[:-1]  # '2014-8-22__14-8'


def make_dir(dir_path):
    """Creates a directory with specified name."""
    # Create a directory, if it exists, ignore the error.
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as exception:
            raise


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = None, args[0] #None and index
    finally:
        signal.alarm(0)

    return result


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nr_queries = 100
ks = range(1, 11) + range(12, 51, 2)
test_ks = range(1, 60, 3)
dir_name = get_dir_name()
test_dir = os.path.join(os.path.join(project_dir, "run-script"), dir_name)
make_dir(test_dir)
queries_dir = "datasets/sample/queries"
make_dir(os.path.join(project_dir, queries_dir))
time_start = time.time()
for name in samples.keys():
    #create folder for this test
    queries_filename = queries_dir+"/"+name+"_queries.npy"
    queries_filename = os.path.join(project_dir, queries_filename)
    dataset_dir = os.path.join(test_dir, name) #folder for this dataset
    make_dir(dataset_dir) #create folder
    ds = load_dataset(name)
    data = ds.data
    if not os.path.isfile(queries_filename):
        #generate queries
        queries = random.sample(ds.data, nr_queries) #random query points from dataset
        queries = np.array(queries)
        np.save(queries_filename, queries)
        print "queries saved! to:", queries_filename
    else:
        queries = np.load(queries_filename)
        print "queries loaded! from:", queries_filename
    
    print "Dataset: %s, dim:%d, examples:%d" % (ds.name, len(ds.data[0]), len(ds.data))
    time_dataset = time.time()
    algorithms = [aa for aa in algorithms_bruteforce]
    if len(ds.data[0]) > 10:
        algorithms += algorithms_approx
    elif len(ds.data[0]) == 10:
        algorithms += algorithms_exact
        #algorithms += algorithms_approx
    else:
        algorithms += algorithms_exact

    pickle_path = os.path.join(project_dir,"datasets/sample/builds/"+name)
    make_dir(pickle_path)
    for algo_name, algo, build_params, query_params in algorithms:
        print "starting algo:", algo_name
        
        try:
            build_time, index = get_build_info(algo_name, algo, build_params, ds, pickle_path)
        except Exception, e:
            print "Failed --> dataset:%s, algorithm:%s" % (name, algo_name)
            print "exception:", e
            continue
        if index is None:
            print "Failed --> dataset:%s, algorithm:%s" % (name, algo_name)
            continue
        else:
            print "starting querying..."
            try:
                time_query = time.time()
                query_times, query_times_time, query_times_all_clock, query_times_all_time, precisions, missings, \
                    checked, dc = get_query_info(index, queries, ks, query_params)
                print "finished --> build_time:%f, query_time:%f" % (build_time, time.time()-time_query)
            
                save_info(algo_name, build_time, query_times, query_times_time, query_times_all_clock, query_times_all_time,
                            precisions, missings, checked, dc, ks, dataset_dir)
            except Exception, e:
                print "Failed --> dataset:%s, algorithm:%s" % (name, algo_name)
                print "exception:", e
                continue
        if not gc.isenabled():
            gc.enable()
        gc.collect()
    print "finished dataset:%s, time needed:%f" % (name, time.time()-time_dataset)

print "finished all, took:", (time.time() - time_start)
