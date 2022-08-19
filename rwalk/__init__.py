"""
The signature of the C function is:
void random_walk(int const* ptr, int const* neighs, int n, int num_walks,
                 int num_steps, int seed, int nthread, int* walks);
"""
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from os.path import dirname

basepath='/Users/gpirro/Documents/PyCharm_WORKSPACE/triple2vec/data/'

datatype=np.int32

array_1d_int = npct.ndpointer(dtype=datatype, ndim=1, flags='CONTIGUOUS')

librwalk = npct.load_library("librwalk", dirname(__file__))

print("Loading Fast Random Walk Library Library from: {}".format(dirname(__file__)))
librwalk.random_walk.restype = None
librwalk.random_walk.argtypes = [array_1d_int, array_1d_int, c_int, c_int, c_int, c_int, c_int, array_1d_int]

librwalk.biased_random_walk.restype = None
librwalk.biased_random_walk.argtypes = [array_1d_int, array_1d_int, array_1d_int, c_int, c_int, c_int, c_int, c_int, array_1d_int]

def biased_random_walk(ptr, neighs, weights, num_walks=10, num_steps=3, nthread=-1, seed=111413):
    assert (ptr.flags['C_CONTIGUOUS'])
    assert (neighs.flags['C_CONTIGUOUS'])
    assert (weights.flags['C_CONTIGUOUS'])

    assert (ptr.dtype == datatype)
    assert (neighs.dtype == datatype)
    assert (weights.dtype == datatype)

    n = ptr.size - 1;

    walks = -np.ones((n * num_walks, (num_steps + 1)), dtype=datatype, order='C')
    assert (walks.flags['C_CONTIGUOUS'])

    librwalk.biased_random_walk(
        ptr,
        neighs,
        weights,
        n,
        num_walks,
        num_steps,
        seed,
        nthread,
        np.reshape(walks, (walks.size,), order='C'))

    return walks

def random_walk(ptr, neighs, num_walks=10, num_steps=3, nthread=-1, seed=111413):
    assert (ptr.flags['C_CONTIGUOUS'])
    assert (neighs.flags['C_CONTIGUOUS'])
    assert (ptr.dtype == datatype)
    assert (neighs.dtype == datatype)

    n = ptr.size - 1;
    walks = -np.ones((n * num_walks, (num_steps + 1)), dtype=datatype, order='C')
    assert (walks.flags['C_CONTIGUOUS'])

    librwalk.random_walk(
        ptr,
        neighs,
        n,
        num_walks,
        num_steps,
        seed,
        nthread,
        np.reshape(walks, (walks.size,), order='C'))

    return walks


def read_weighted_edgelist(fname, comments='#'):
    edges = np.genfromtxt(fname, comments=comments, delimiter=",",
                          defaultfmt='%d',dtype=np.float)

    edge_weights=np.array(edges[:,2],dtype=np.float)

    edges=np.array(edges[:,:2],dtype=datatype)
    edges=edges[1:]
    #print(len(edges[1:]) #12933204

    assert (len(edges.shape) == 2)
    assert (edges.shape[1] == 2)

    # read weights and put them in a separate structure
    # we need to pass to the c code this additional vector
    # normalize each row weights to sum 1 (turn into  probabilities of picking an edge)
    # Sort so smaller index comes first
    edges.sort(axis=1)
    # This is the TOTAL number of nodes (assuming nodes, in edges, are numbered from 0)
    n = (np.amax(edges) + 1)

    #n=5516361
    # Duplicate
    duplicated_edges = np.vstack((edges, edges[:, ::-1]))
    #print(duplicated_edges)

    # Sort duplicated edges by first index
    _tmp = np.zeros((duplicated_edges.shape[0],), dtype=datatype)
    _tmp += duplicated_edges[:, 0]
    _tmp *= np.iinfo(datatype).max
    _tmp += duplicated_edges[:, 1]

    ind_sort = np.argsort(_tmp)
    sorted_edges = duplicated_edges[ind_sort, :]

    # THIS IS THE CSR format: https://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf page 19
    # Calculate degree and create ptr, neighs
    # sorted_edges[:, 0] counts the number targets for each source
    vals, counts = np.unique(sorted_edges[:, 0], return_counts=True)

    # Degree of each node
    degs = np.zeros((n,))

    #print(len(degs))

    # assign the degree to each node
    degs[vals] = counts

    #
    ptr = np.zeros((n + 1,),dtype=datatype)

    # vals= [0    1    2... 1997 1998 1999]
    # degs [23 20 17... 22 10 12]
    # ptr [0    23    43... 40106 40116 40128]

    # RECALL edges are sorted, the start is the smaller number
    ptr[1:] = np.cumsum(degs)

    # sorted_edges[:, 1] get the target of each edge
    # NEIGHS contains the TARGET of each edge!
    neighs = np.copy(sorted_edges[:, 1])

    # using ptr one can do:
    # num_neighs = ptr[curr + 1] - ptr[curr] that returns
    # the num of neighbours for the current node

    # Check ptr, neighs, weights
    ptr.flags.writeable = False
    assert (ptr.flags.owndata == True)
    assert (ptr.flags.aligned == True)
    assert (ptr.flags.c_contiguous == True)

    neighs.flags.writeable = False
    assert (neighs.flags.owndata == True)
    assert (neighs.flags.aligned == True)
    assert (neighs.flags.c_contiguous == True)

    edge_weights.flags.writeable = False
    assert (edge_weights.flags.owndata == True)
    assert (edge_weights.flags.aligned == True)
    assert (edge_weights.flags.c_contiguous == True)

    return ptr, neighs, edge_weights
