import pickle
import numpy as np
import scipy
import scipy.spatial.distance as distance
from numpy import sqrt

def _convert_to_double(X):
  if X.dtype != np.double:
    X = X.astype(np.double)
  if not X.flags.contiguous:
    X = X.copy()
  return X

def get_indices(k,n=31085):
  #length = total-1
  #row = -1
  #while residue > 0:
  #  residue -= length
  #  row += 1
  #  length -= 1
  #col = total + residue
  #return row%total, col%total
  i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
  j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
  return i, j

total_links = 1

bible_path = '/users/grads/pra3/Desktop/pra3/aml/scratch/bibles/tandem1.pickle'
distance_path = '/users/grads/pra3/Desktop/pra3/Documents/cosine_distances.pickle'
dump_path = '/users/grads/pra3/Desktop/pra3/Documents/adjacencies/adjacency_matrix'+str(total_links)+'.pickle'

bibletext = pickle.load(open(bible_path,"rb"))

#dists = pickle.load(open(distance_path,"rb"))
dists = []
for i in bibletext.documents:
  dists.append(i.metadata['tandem1_theta'])

dists = np.copy(dists)
method = 'single'
metric = 'cosine'
dists = _convert_to_double(np.asarray(dists,order='c'))
optimal_ordering = False
print("All loaded")

_LINKAGE_METHODS = {'single':0,'complete':1,'average':2,'centroid':3,'median':4,'ward':5,'weighted':6}
_EUCLIDEAN_METHODS = {'centroid','median','ward'}

dists = distance.pdist(dists,metric)
print("dists complete")

indices = np.argsort(dists,axis=None)
print("dists sorted")

distances = []
adjacency_matrix = []



for i in range(len(bibletext.documents)):
  distances.append(2)
  adjacency_matrix.append([])
i=0
for index in indices:
  i+=1
  if i%10000000==0:
    print(i//10000000)
  row,col = get_indices(index)
  if dists[int(index)] <= distances[int(row)] or len(adjacency_matrix[row]) < total_links:
    distances[int(row)] = dists[int(index)]
    adjacency_matrix[row].append(col)
  if dists[int(index)] <= distances[int(col)] or len(adjacency_matrix[int(col)]) < total_links:
    distances[int(col)] = dists[int(index)]
    adjacency_matrix[int(col)].append(row)

pickle.dump(adjacency_matrix,open(dump_path,"wb"))
