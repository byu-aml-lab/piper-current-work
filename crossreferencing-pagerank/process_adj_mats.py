import glob
import os
import networkx as nx
import pickle
import numpy as np
from numpy import sqrt

def get_linear_index(i,j,n=31085):
  k = (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
  return int(k)

def get_triu_indices(k,n=31085):
  i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
  j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
  return int(i),int(j)

def get_verse(index):
  return bible.documents[index].metadata['verse']

bible = pickle.load(open("/users/grads/pra3/Desktop/pra3/aml/scratch/bibles/tandem1.pickle","rb"))

cos_dists = pickle.load(open("/users/grads/pra3/Desktop/pra3/Documents/cosine_distances.pickle","rb"))

fnames = glob.glob('/users/grads/pra3/Desktop/pra3/Documents/adjacencies/*.pickle')

for fname in fnames:
  name = os.path.basename(fname)[:-7]
  #if name=="adjacency_matrix" or name=="adjacency_matrix2":
  #  continue  
  adj_mat = pickle.load(open(fname,"rb"))
  #tam_jda = []
  #for i in range(31084):
  #  tam_jda.append([])

  #for node,connections in enumerate(adj_mat):
  #  for dest in connections:
  #    tam_jda[int(dest)].append(int(node))

  G = nx.DiGraph()
  for node,temp in enumerate(adj_mat):
    for dest in temp:
      G.add_edge(node,int(dest))

  pr = nx.pagerank(G)

  sortme = []
  for i in range(31085):
    sortme.append(pr[i])

  done = np.argsort(sortme)
  done = np.flip(done)

  minimum = sortme[done[-1]]

  cat_dists = []

  for text,index in enumerate(done):
    #if sortme[index] <= minimum:
    #  break
    curdists = []
    for i in range(31085):
      if i is not index:
        if i < index:
          k = get_linear_index(i,index)
        else:
          k = get_linear_index(index,i)
        curdists.append(cos_dists[k])
    sorted_curdists = np.argsort(curdists)
    cat_dists.append(sorted_curdists)

  cat_dists = np.array(cat_dists)

  checked = np.zeros((31085,31085),np.bool_)
  for i in range(31085):
    checked[i][i] = True
  with open('/users/grads/pra3/Desktop/pra3/Documents/xrefs/{}_undirected.txt'.format(name),'w') as f:
    for i in range(cat_dists.shape[1]):
      for j in range(cat_dists.shape[0]):
        if sortme[done[j]] <= minimum:
            break
        index = done[j]
        dist_index = cat_dists[j][i]
        if not checked[index][dist_index] and not checked[dist_index][index]:
          checked[index][dist_index]=True
          checked[dist_index][index]=True
          f.write('{} {}\n'.format(get_verse(index),get_verse(dist_index)))
          f.write('{} {}\n'.format(get_verse(dist_index),get_verse(index)))
  name=None
  adj_mat=None
  G = None
  pr = None
  sortme=None
  done=None
  minimum=None
  cat_dists = None
  curdists = None
  sorted_curdists = None
  checked = None
  f.close()
  f = None

