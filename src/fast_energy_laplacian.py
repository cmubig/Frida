'''
NOTE: This code came from recovery.py, which can be found on GitHub:
        https://github.com/yig/harmonic_interpolation
'''



from numpy import *

def gen_symmetric_grid_laplacian2( rows, cols, cut_edges = None ):
    '''
    The same as 'gen_symmetric_grid_laplacian1()', except boundary weights are correct.
    
    tested
    (see also test_cut_edges())
    
    '''
    
    assert rows > 0
    assert cols > 0
    
    
    from scipy import sparse
    
    N = rows
    M = cols
    def ind2ij( ind ):
        assert ind >= 0 and ind < N*M
        return ind // M, ind % M
    def ij2ind( i,j ):
        assert i >= 0 and i < N and j >= 0 and j < M
        return i*M + j
    
    Adj = []
    AdjValues = []
    
    ## The middle (lacking the first and last columns) strip down
    ## to the bottom, not including the bottom row.
    for i in range( 0, rows-1 ):
        for j in range( 1, cols-1 ):
            
            ind00 = ij2ind( i,j )
            indp0 = ij2ind( i+1,j )
            Adj.append( ( ind00, indp0 ) )
            AdjValues.append( .25 )
    
    ## The first and last columns down to the bottom,
    ## not including the bottom row.
    for i in range( 0, rows-1 ):
        for j in ( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            indp0 = ij2ind( i+1,j )
            Adj.append( ( ind00, indp0 ) )
            AdjValues.append( .125 )
    
    ## The middle (lacking the first and last rows) strip to
    ## the right, not including the last column.
    for i in range( 1, rows-1 ):
        for j in range( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            ind0p = ij2ind( i,j+1 )
            Adj.append( ( ind00, ind0p ) )
            AdjValues.append( .25 )
    
    ## The first and last rows over to the right,
    ## not including the right-most column.
    for i in ( 0, rows-1 ):
        for j in range( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            ind0p = ij2ind( i,j+1 )
            Adj.append( ( ind00, ind0p ) )
            AdjValues.append( .125 )
    
    ## Build the adjacency matrix.
    AdjMatrix = sparse.coo_matrix( ( AdjValues, asarray( Adj ).T ), shape = ( rows*cols, rows*cols ) )
    # AdjMatrix = sparse.coo_matrix( ( AdjValues, Adj  ), shape = ( 1, 1 ) )
    ## We have so far only counted right and downward edges.
    ## Add left and upward edges by adding the transpose.
    AdjMatrix = AdjMatrix.T + AdjMatrix
    #AdjMatrix = AdjMatrix.tocsr()
    

    # if cut_edges is None: cut_edges = []
    
    # # assert_valid_cut_edges( rows, cols, cut_edges )
    # ## Build the adjacency matrix representing cut edges and subtract it
    # if len( cut_edges ) > 0:
    #     CutAdj = []
    #     for ij, kl in cut_edges:
    #         CutAdj.append( ( ij2ind( *ij ), ij2ind( *kl ) ) )
    #         CutAdj.append( ( ij2ind( *kl ), ij2ind( *ij ) ) )
    #     CutAdjMatrix = sparse.coo_matrix( ( ones( len( CutAdj ) ), asarray( CutAdj ).T ), shape = ( rows*cols, rows*cols ) )
        
    #     ## Update AdjMatrix.
    #     ## We need to subtract the component-wise product of CutAdjMatrix and AdjMatrix
    #     ## because AdjMatrix has non-zero values and CutAdjMatrix acts like a mask.
    #     AdjMatrix = AdjMatrix - CutAdjMatrix.multiply( AdjMatrix )
    
    '''
    ## One over mass
    ooMass = sparse.identity( rows*cols )
    ooMass.setdiag( 1./asarray(AdjMatrix.sum(1)).ravel() )
    ## NOTE: ooMass*AdjMatrix isn't symmetric because of boundaries!!!
    L = sparse.identity( rows*cols ) - ooMass * AdjMatrix
    '''
    
    ## This formulation is symmetric: each vertex has a consistent weight
    ## according to its area (meaning boundary vertices have smaller
    ## weights than interior vertices).
    ## NOTE: I tried sparse.dia_matrix(), but sparse.dia_matrix.setdiag() fails with a statement that dia_matrix doesn't have element assignment.
    ## UPDATE: setdiag() seems to just be generally slow.  coo_matrix is fast!
    #Mass = sparse.lil_matrix( ( rows*cols, rows*cols ) )
    #Mass.setdiag( asarray(AdjMatrix.sum(1)).ravel() )
    #debugger()
    Mass = sparse.coo_matrix( ( asarray(AdjMatrix.sum(1)).ravel(), ( range( rows*cols ), range( rows*cols ) ) ) )
    L = ( Mass - AdjMatrix )
    
    ## The rows should sum to 0.
    assert ( abs( asarray( L.sum(1) ).ravel() ) < 1e-5 ).all()
    ## The columns should also sum to 0, since L is symmetric.
    assert ( abs( asarray( L.sum(0) ).ravel() ) < 1e-5 ).all()
    ## It should be symmetric.
    assert len( ( L - L.T ).nonzero()[0] ) == 0
    
    return L.tocsr()

gen_grid_laplacian = gen_symmetric_grid_laplacian2
