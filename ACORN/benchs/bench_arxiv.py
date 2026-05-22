# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import numpy as np
import re
import faiss
from datetime import datetime
from multiprocessing.pool import ThreadPool
from datasets import ivecs_read


# Logger class for dual output (console + file)
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        self.log.write(f"Benchmark started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"="*70 + "\n\n")
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.write(f"\n{'='*70}\n")
        self.log.write(f"Benchmark ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.close()


# we mem-map the biggest files to avoid having them in memory all at
# once


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


#################################################################
# Bookkeeping
#################################################################


dbname        = sys.argv[1]
index_key     = sys.argv[2]
parametersets = sys.argv[3:]


tmpdir = '/tmp/bench_polysemous'

if not os.path.isdir(tmpdir):
    print("%s does not exist, creating it" % tmpdir)
    os.mkdir(tmpdir)


#################################################################
# Prepare dataset
#################################################################


print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    xb = mmap_bvecs('bigann/bigann_base.bvecs')
    xq = mmap_bvecs('bigann/bigann_query.bvecs')
    xt = mmap_bvecs('bigann/bigann_learn.bvecs')

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]

    gt = ivecs_read('bigann/gnd/idx_%dM.ivecs' % dbsize)

elif dbname == 'Deep1B':
    xb = mmap_fvecs('deep1b/base.fvecs')
    xq = mmap_fvecs('deep1b/deep1B_queries.fvecs')
    xt = mmap_fvecs('deep1b/learn.fvecs')
    # deep1B's train is is outrageously big
    xt = xt[:10 * 1000 * 1000]
    gt = ivecs_read('deep1b/deep1B_groundtruth.ivecs')

elif dbname == 'arxiv' or dbname == 'arxiv_all':
    # Load arxiv dataset - base data is shared
    xb = mmap_fvecs('../../../datasets/discrete/arxiv/arxiv_base.fvecs')
    xt = xb[:100000]  # Use first 100k vectors for training
    
    # Determine which query types to test
    if dbname == 'arxiv_all':
        query_types = ['equal', 'or', 'and']
    else:
        query_types = ['equal']  # default backward compatibility
    
    # Store all query types data
    queries_data = {}
    for qtype in query_types:
        xq_temp = mmap_fvecs(f'../../../datasets/discrete/arxiv/arxiv_query_{qtype}.fvecs')
        gt_temp = np.loadtxt(f'../../../datasets/discrete/arxiv/arxiv_gt_{qtype}.txt', dtype='int32')
        queries_data[qtype] = {'xq': xq_temp, 'gt': gt_temp}
    
    # Use first query type for initial setup (will loop through all later)
    first_type = query_types[0]
    xq = queries_data[first_type]['xq']
    gt = queries_data[first_type]['gt']

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)


print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq


#################################################################
# Training
#################################################################


def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 256 * 1000

    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall('IMI2x([0-9]+)', index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return n_train


def get_trained_index():
    filename = "%s/%s_%s_trained.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key)

        n_train = choose_train_size(index_key)

        xtsub = xt[:n_train]
        print("Keeping %d train vectors" % xtsub.shape[0])
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print("train done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Adding vectors to dataset
#################################################################

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda i01: x[i01[0]:i01[1]].astype('float32').copy(),
        block_ranges)


def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Perform searches
#################################################################

# Create results directory and log file
results_dir = 'results'
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{results_dir}/bench_{dbname}_{index_key.replace(',', '_')}_{timestamp}.txt"

# Initialize logger
logger = Logger(log_filename)
sys.stdout = logger

print(f"Results will be saved to: {log_filename}")
print(f"Dataset: {dbname}")
print(f"Index: {index_key}")
print(f"Parameters: {parametersets}")
print()

index = get_populated_index()

ps = faiss.ParameterSpace()
ps.initialize(index)

# Determine which query types to test
if dbname == 'arxiv_all':
    test_query_types = query_types
else:
    test_query_types = [query_types[0]] if 'query_types' in locals() else ['single']
    if 'queries_data' not in locals():
        queries_data = {'single': {'xq': xq, 'gt': gt}}

# Loop through each query type
for qtype in test_query_types:
    print(f"\n{'='*70}")
    print(f"Testing query type: {qtype.upper()}")
    print(f"{'='*70}\n")
    
    # Load corresponding query and ground truth
    xq = queries_data[qtype]['xq'].astype('float32').copy()
    gt = queries_data[qtype]['gt']
    nq = xq.shape[0]
    
    # a static C++ object that collects statistics about searches
    ivfpq_stats = faiss.cvar.indexIVFPQ_stats
    ivf_stats = faiss.cvar.indexIVF_stats

    if parametersets == ['autotune'] or parametersets == ['autotuneMT']:

        if parametersets == ['autotune']:
            faiss.omp_set_num_threads(1)

        # setup the Criterion object: optimize for 1-R@1
        crit = faiss.OneRecallAtRCriterion(nq, 1)
        # by default, the criterion will request only 1 NN
        crit.nnn = 100
        crit.set_groundtruth(None, gt.astype('int64'))

        # then we let Faiss find the optimal parameters by itself
        print("exploring operating points")

        t0 = time.time()
        op = ps.explore(index, xq, crit)
        print("Done in %.3f s, available OPs:" % (time.time() - t0))

        # opv is a C++ vector, so it cannot be accessed like a Python array
        opv = op.optimal_pts
        print("%-40s  1-R@1     time" % "Parameters")
        for i in range(opv.size()):
            opt = opv.at(i)
            print("%-40s  %.4f  %7.3f" % (opt.key, opt.perf, opt.t))

    else:

        # we do queries in a single thread
        faiss.omp_set_num_threads(1)

        print(' ' * len(parametersets[0]), '\t', 'R@1    R@10   R@100     time    %pass')

        for param in parametersets:
            print(param, '\t', end=' ')
            sys.stdout.flush()
            ps.set_index_parameters(index, param)
            t0 = time.time()
            ivfpq_stats.reset()
            ivf_stats.reset()
            D, I = index.search(xq, 100)
            t1 = time.time()
            for rank in 1, 10, 100:
                n_ok = (I[:, :rank] == gt[:, :1]).sum()
                print("%.4f" % (n_ok / float(nq)), end=' ')
            print("%8.3f  " % ((t1 - t0) * 1000.0 / nq), end=' ')
            print("%5.2f" % (ivfpq_stats.n_hamming_pass * 100.0 / ivf_stats.ndis))

# Close logger and restore stdout
print(f"\nResults saved to: {log_filename}")
logger.close()
sys.stdout = logger.terminal
