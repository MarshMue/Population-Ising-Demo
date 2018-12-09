from scipy.sparse import identity
import numpy as np

# the inner product expression as defined in the CAPY paper, explicitly counts the 
# number of edges between different populations and is approximately 2 times the explicit number of edges between the same population
def inner_product(x, y, A):
    '''
    x - population vector for x
    y - population vector for y
    A - adjacency matrix for graph
    
    return - energy score of cluster
    '''
    result = np.matmul(x.T, np.matmul(A + np.identity(A[0].size), y))
    return result[0,0]

# the inner product expression as defined in the CAPY paper, explicitly counts the 
# number of edges between different populations and is approximately 2 times the explicit number of edges between the same population
# computations done with sparse matrices for significant speedup
def sparse_inner_product(x, y, A):
    '''
    x - population vector for x
    y - population vector for y
    A - sparse adjacency matrix for graph
    
    return - energy score of cluster
    '''
    M = A + identity(A.shape[0])
    return np.dot(x.T, M.dot(y))[0,0]

# the half edge capy score, interpretible as the average of the probabilities of the type: if I'm a node of type X, what is the probability my neighbor is type X also?
def half_capy(x,y,A):
    '''
    x - population vector for x
    y - population vector for y
    A - adjacency matrix for graph
    
    return - the half-edge capy score
    '''
    xx = inner_product(x,x,A)
    xy = inner_product(x,y,A)
    yy = inner_product(y,y,A)
    return (xx/(xx+xy) + yy/(yy+xy))/2

# the edge capy score, interpretible as the average of probabilities of the type: given that i pick an edge that touches a node of type X, what is the probability it connects two nodes of type X?
def edge_capy(x,y,A):
    '''
    x - population vector for x
    y - population vector for y
    A - adjacency matrix for graph

    return - the edge capy score
    '''
    xx = inner_product(x,x,A)
    xy = inner_product(x,y,A)
    yy = inner_product(y,y,A)
    return (xx/(xx+2*xy) + yy/(yy+2*xy))/2


# the half edge capy score, done with sparse matrices for faster computation
def sparse_half_capy(x,y,A):
    '''
    x - population vector for x
    y - population vector for y
    A - sparse adjacency matrix for graph
    
    return - the half-edge capy score
    '''
    xx = sparse_inner_product(x,x,A)
    xy = sparse_inner_product(x,y,A)
    yy = sparse_inner_product(y,y,A)
    return (xx/(xx+xy) + yy/(yy+xy))/2

def sparse_edge_capy(x,y,A):
    '''
    x - population vector for x
    y - population vector for y
    A - sparse adjacency matrix for graph
    
    return - the edge capy score
    '''
    xx = sparse_inner_product(x,x,A)
    xy = sparse_inner_product(x,y,A)
    yy = sparse_inner_product(y,y,A)
    return (xx/(xx+2*xy) + yy/(yy+2*xy))/2