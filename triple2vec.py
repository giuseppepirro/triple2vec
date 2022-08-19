'''
Created on Oct 22, 2019
Implementation of triple2vec in:
V. Fionda, G. Pirrò Learning Triple Embeddings from Knowledge Graphs. AAAI 2020
@author: Giuseppe Pirrò (pirro@di.uniroma1.it)

Please cite the paper if you  use it
'''

import pandas as pd
import csv as csv
import numpy as np
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Set
import pickle
import os,sys,inspect
from joblib import Parallel, delayed
import multiprocessing
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import scipy.sparse as ss
import scipy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0, './rwalk')

import rwalk

basepath='./data/'
num_cores = multiprocessing.cpu_count()

def get_triple_dictionary(filename, KG):
    plk_file=basepath + KG + '/'+filename+'.pkl'
    if os.path.isfile(plk_file):
        print("Reading triple2ids from pkl...")
        dict_data = pickle.load(open(plk_file, 'rb'))
        return dict_data
    else:
        print("Creating triple dictionary.....")
        construct_triple_dictionary(KG)
        csv_file = basepath + KG + '/' + filename + '.csv'
        data =csv.DictReader(open(csv_file),fieldnames=('Triple', 'ID'))
        dict_data = {line['Triple']:line['ID'] for line in data}
    output = open(plk_file,'wb')
    pickle.dump(dict_data, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()
    return dict_data

def construct_csv_file(filenameIn, filenameOut,KG):
    df = pd.read_csv(basepath+KG+'/'+filenameIn, delimiter='	',header=None,usecols=[0,1,2])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.rename(columns={'2': 's', '0': 'o','1': 'p'}, inplace=True)
    df.to_csv(basepath+KG+'/'+filenameOut,index=False)


def construct_all_triples_file(data_file,KG):
    dfTrain = pd.read_csv(basepath+KG+"/"+data_file, delimiter=',')
####Predicate dictionary
    dict_p = set(pd.Series(dfTrain.p.values, index=dfTrain.p).to_dict())
    dict_Preds = dict({item: val for val, item in enumerate(dict_p)})
    with open(basepath+KG+'/preds2ids.csv',
              'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["pred", "id"])
        for key, value in dict_Preds.items():
            writer.writerow([key, value])
####Entities dictionary
    dict_s = set(pd.Series(dfTrain.s.values, index=dfTrain.s).to_dict())
    dict_o = set(pd.Series(dfTrain.o.values, index=dfTrain.o).to_dict())
    dict_e=set(dict_s.union(dict_o))
    dict_Entities = dict({item: val for val, item in enumerate(dict_e)})
    with open(basepath+KG+'/entities2ids.csv',
              'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["entity", "id"])
        for key, value in dict_Entities.items():
            writer.writerow([key, value])
    print("Start replacing....")
    dfTrain.p = dfTrain.p.map(lambda x: dict_Preds.get(x, x))
    print("End replacing predicate....")
    dfTrain.s = dfTrain.s.map(lambda x: dict_Entities.get(x, x))
    print("End replacing subject....")
    dfTrain.o = dfTrain.o.map(lambda x: dict_Entities.get(x, x))
    print("End replacing object....")
    dfTrain.to_csv(basepath+KG+'/triples.csv',
        index=False)

def convert_weighted_line_graph_data_to_pickle (ptr, neighs, edge_weights,KG):
    data=[ptr, neighs, edge_weights]
    output = open(basepath + KG + '/'+'line_graph'+'.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

def convert_csv_to_pickle (filename,KG):
    data = pd.read_csv(basepath+KG+'/'+filename+'.csv', delimiter=',')
    output = open(basepath + KG + '/'+filename+'.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

def construct_triple_dictionary(KG):
    triples = pd.read_csv(basepath+KG+'/triples.csv', delimiter=',')
    vals=range(0,len(triples))
    dict_triples = (pd.Series(data=zip(triples.s, triples.o,triples.p), index=vals).to_dict())
    with open(basepath + KG + '/triples2ids.csv',
            'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["triple", "id"])
        for key, value in dict_triples.items():
            writer.writerow([str(value).replace(' ', ''), key])

def compute_relatedness_matrix(KG):
    start=time()
    print("Computing relatedness matrix for "+KG)
    path_to_data = basepath+KG+'/triples.csv'
    df = pd.read_csv(path_to_data, delimiter=',')
    groups = df.groupby('p')
    n_preds = len(groups)
    rel_matrix = np.zeros((n_preds, n_preds))
    tf = [(np.log(1 + pd.merge(groups.get_group(j), groups.get_group(i), how='right', on=['s', 'o']).dropna().shape[0]))
            for i in range(0, n_preds) for j in range(0, n_preds)]
    pos = [(i, j) for i in range(0, n_preds) for j in range(0, n_preds)]
    rows, cols = zip(*pos)
    rel_matrix[rows, cols] = tf
    idf_values={i: np.log(n_preds/np.count_nonzero(rel_matrix[i])) for i in range(0, n_preds)}
    tf_idf=[rel_matrix[i][j]*idf_values[j] for i in range(0, n_preds) for j in range(0, n_preds)]
    rel_matrix[rows, cols] = tf_idf
    similarity=[cosine_similarity(rel_matrix[i].reshape(1,n_preds),rel_matrix[j].reshape(1,n_preds))[0][0]+0.001 for i in range(0, n_preds) for j in range(0, n_preds)]
    rel_matrix[rows, cols] = similarity
    np.savetxt(basepath+KG+"/rel_matrix.csv", rel_matrix, fmt='%1.3f',delimiter=",")

    print("Time to compute relatedness matrix for {} {:.6f} s".format(KG,(time() - start)))

    return rel_matrix

def get_relatedness_matrix(KG):
    matrix_file = basepath + KG + '/rel_matrix.csv'
    if os.path.isfile(matrix_file):
        print("Reading relatedness matrix...")
        relatedness_matrix = np.loadtxt(os.path.join(matrix_file), delimiter=",");
        relatedness_matrix = np.array(relatedness_matrix).astype("float")
        return relatedness_matrix
    else:
        return compute_relatedness_matrix(KG)

def get_knowledge_graph(KG):
    plk_file = basepath + KG + '/triples.pkl'
    if os.path.isfile(plk_file):
        print("Loading triple pkl file.....")
        result = pickle.load(open(plk_file, 'rb'))
        return result[0],result[1],result[2],result[3]
    else:
        print("Creating triple pkl file.....")
        edges = np.genfromtxt(basepath+KG+'/triples.csv', delimiter=",",comments='#',
                              defaultfmt='%d',dtype=np.int32)

        #predicates
        edge_predicates=np.array(edges[:,2],dtype=np.int32)

        #pairs (subj,obj)
        edges=np.array(edges[:,[0, 1]],dtype=np.int32)

        n = (np.amax(edges) + 1)

        # THIS IS THE CSR format: https://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf page 19
        # Calculate degree and create ptr, neighs
        # sorted_edges[:, 0] counts the number targets for each source
        ## we are now using edges instead of sorted edges
        vals, counts = np.unique(edges[:, 0], return_counts=True)

        # Degree of each node
        degs = np.zeros((n,))

        # assign the degree to each node
        degs[vals] = counts
        #
        ptr = np.zeros((n + 1,),dtype=np.int32)

        # RECALL edges are sorted, the start is the smaller number
        ptr[1:] = np.cumsum(degs)
        # sorted_edges[:, 1] get the target of each edge
        # NEIGHS contains the TARGET of each edge!
        neighs = np.copy(edges[:, 1])

        # using ptr one can do:
        # num_neighs = ptr[curr + 1] - ptr[curr] that returns
        # the num of neighbours for the current node

        #save data into pickle format
        data = [ptr, neighs, edges, edge_predicates]
        output = open(basepath + KG + '/' + 'triples' + '.pkl', 'wb')
        pickle.dump(data, output)
        output.close()

        print("#Nodes: {}, #Edges {}".format(len(ptr),len(edges)))
        return ptr, neighs, edges, edge_predicates

def read_weighted_line_graph(KG):
    plk_file = basepath + KG + '/line_graph.pkl'
    csv_file=basepath + KG + '/line_graph.csv'
    if os.path.isfile(plk_file):
        print("Loading line graph pkl file.....")
        result = pickle.load(open(plk_file, 'rb'))
        return result[0], result[1], result[2]
    else:
        edges = np.genfromtxt(csv_file, comments="#", delimiter=",",
                              defaultfmt='%d',dtype=np.float)
        datatype = np.int32
        edge_weights=np.array(edges[:,2],dtype=np.float)
        edges=np.array(edges[:,:2],dtype=datatype)
        edges=edges[1:]
        assert (len(edges.shape) == 2)
        assert (edges.shape[1] == 2)

        n = (np.amax(edges) + 1)

        # Duplicate
        duplicated_edges = np.vstack((edges, edges[:, ::-1]))

        # Sort duplicated edges
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

        # assign the degree to each node
        degs[vals] = counts

        #
        ptr = np.zeros((n + 1,),dtype=datatype)

        # vals= [0    1    2... 1997 1998 1999]
        # degs [23 20 17... 22 10 12]
        # ptr [0    23    43... 40106 40116 40128]

        # RECALL edges are sorted, the start is the smaller number
        ptr[1:] = np.cumsum(degs)

        # NEIGHS contains the TARGET of each edge!
        neighs = np.copy(sorted_edges[:, 1])

        # using ptr one can do:
        # num_neighs = ptr[curr + 1] - ptr[curr] that returns
        # the num of neighbours for the current node

        data = [ptr, neighs, edge_weights]
        output = open(basepath + KG + '/' + 'line_graph' + '.pkl', 'wb')
        pickle.dump(data, output)
        output.close()

    return ptr, neighs, edge_weights


def compute_triple_line_graph(KG):
    plk_file = basepath + KG + '/line_graph.pkl'
    if os.path.isfile(plk_file):
        print("Line graph already exists .....")
        return
    else:
        start = time()
        ptr, neighs, edges, edge_predicates = get_knowledge_graph(KG)
        print("Time to read KG {:.6f} s".format(time() - start))

        start = time()
        triple2ids_dict=get_triple_dictionary('triples2ids', KG)
        print("Time to read triple dictionary {:.6f} s".format((time() - start)))

        start = time()
        rel_matrix=get_relatedness_matrix(KG)
        print("Time to read relatedness matrix {:.3f} s".format((time() - start)))

        results = Parallel(n_jobs=num_cores+2,verbose=10,prefer="threads")(delayed(add_edges)(e, edges, edge_predicates,
                                                                                              pred,triple2ids_dict,rel_matrix) for e, pred in zip(edges[1:],edge_predicates[1:]))
        print("Time to compute line graph {:.6f} s".format(time() - start))

        ###write line graph on disk as pkl and csv
        write_triple_line_graph(results,KG)

#Read the line graph and represents it as a csr matrix
# that will be fed to the code to compute random walks
def get_line_graph_as_sparse_matrix(KG):
    plk_file = basepath + KG + '/line_graph_sparse.pkl'
    if os.path.isfile(plk_file):
        print("Loading line triple graph sparse pkl file.....")
        result = pickle.load(open(plk_file, 'rb'))
        #matrix, ptr, neighs, weights
        return result[0], result[1], result[2], result[3]
    else:
        csv_file = basepath + KG + '/line_graph.csv'
        "Read data file and return sparse matrix in coordinate format."
        data=np.genfromtxt(csv_file,  delimiter=",",dtype=np.int32)
        rows = data[:,0]
        cols = data[:,1]
        weights = data[:, 2]


        rows_i = data[:, 1]
        cols_i = data[:, 0]
        rows=np.concatenate((rows,rows_i))
        cols=np.concatenate((cols,cols_i))
        ones = np.ones(len(rows), np.uint32)

        matrix = ss.csr_matrix((ones, (rows, cols)))
        data_pkl = [matrix, matrix.indptr, matrix.indices, weights]
        output = open(plk_file, 'wb')
        pickle.dump(data_pkl, output)
        output.close()

    return matrix, matrix.indptr,matrix.indices, weights

#get all edges having edge[1] as target
# This is useful to find the neighbours of triples
# as those triples that share either the subject or the object
def get_edges(edge,edges,edge_predicates):
    rows=np.where(edges[:, 1] == edge[1])
    result = edges[rows]
    preds = edge_predicates[rows]
    return result, preds

## This is called by the various threads
def add_edges(e, edges, edge_predicates,pred,triple2ids_dict,rel_matrix)-> Set[Tuple]:
    source_to_lookup="("+str(e[0])+","+str(e[1])+","+str(pred)+")"
    source_node = triple2ids_dict[source_to_lookup]
    edgesAdded: Set[Tuple]=[]
    to_nodes, preds = get_edges(e, edges, edge_predicates)
    for to_node_pred in zip(to_nodes, preds):
        if (str(e) != str(to_node_pred[0])):
            target_to_lookup = "(" + str(to_node_pred[0][0]) + "," + str(to_node_pred[0][1]) + "," + str(to_node_pred[1]) + ")"
            target_node = triple2ids_dict[target_to_lookup]
            rel_value=rel_matrix[int(to_node_pred[1])][int(pred)]
            edge: Tuple[str, str,float]=((source_node),(target_node),rel_value)
            edgesAdded.append(edge)
           # print("Triple line graph pred1={} pred2={} ({},{},{:.3f})".format(to_node_pred[1],pred, source_node,target_node,rel_value))
    return edgesAdded

def write_triple_line_graph(results,KG):
    plk_file = basepath + KG + '/line_graph.pkl'
    res_final = []
    if os.path.isfile(plk_file):
        print("Line graph in pkl format already available... EXIT!!")
        exit()
    else:
        print("Writing triple line graph......")
        csv_f = basepath + KG + '/line_graph.csv'
        with open(csv_f,'w') as csv_file:
            writer = csv.writer(csv_file)
            for res in results:
                for r in res:
                    writer.writerow(r)
                    res_final.append(r)
        output = open(plk_file, 'wb')
        pickle.dump(res_final, output)
        output.close()

def fast_walk_computation(num_walks_per_node, length, KG):
    seed = 111413

    matrix, ptr, neighs, weights=get_line_graph_as_sparse_matrix(KG)

    n_cpus = np.arange(1, multiprocessing.cpu_count() + 1)
    ww = []
    start = time()
    for i, nthread in enumerate(n_cpus):
        walks = rwalk.random_walk(ptr, neighs, num_walks=num_walks_per_node, num_steps=length, nthread=nthread, seed=seed + i)
        ww.append(walks)
    n_walks = len(ww) * num_walks_per_node * len(ptr)
    print("Time to compute {} walks in C  {:.6f} s".format(n_walks,(time() - start)))

    # each ww[i] contains a walk per node as a ndarray originally!
    # each walk inside ww[i] is a list of lenght walk_lenght
    #conver all elements in a walk to string:list(map(str,ww.tolist()))
    final_walks=[list(map(str,ww.tolist())) for ww in walks]

    return final_walks

### Compute the embeddgins given the walks
def computeTripleEmbeddings(walks, KG, embedding_size, context_window):
    start = time()
    print("Start computing triple embeddings....")
    model = Word2Vec(walks, size=embedding_size, window=context_window, min_count=0, sg=1, workers=6, iter=5)
    node_vectors = model.wv
    embedding_file_name = basepath+KG +"/triple_embeddings.kv"
    node_vectors.save(get_tmpfile(embedding_file_name))
    print("Time to compute triple embeddings {:.6f}s".format(time() - start))

def main():

    KG = "DBLP"
    
    # STEP 1 TO BE DONE IF only the triples are available (no dictionaries)
    #tripleFileName="triples.csv"
    #construct_all_triples_file(tripleFileName,KG)
    ##

    ##STEP 2
    # The number of walks is given by num_threads*num_nodes*walks_per_node
    walks_per_node=1
    walk_length=100

    #Embedding parameters
    embedding_size = 64
    context_window = 10

    # line triple graph
    compute_triple_line_graph(KG)

    ##get the walks from the file of the line graph and the graph
    walks=fast_walk_computation(walks_per_node, walk_length, KG)

    #compute embeddings
    computeTripleEmbeddings(walks, KG, embedding_size, context_window)

if __name__ == "__main__":
    main()

