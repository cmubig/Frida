from numpy import *
import itertools
import PIL.Image as Image
import os,sys
import errno
import time
from pprint import pprint
import fast_energy_laplacian
import scipy.sparse
import scipy.optimize
from Extract_PD_layers import *



def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

## Normally it is bad practice to put a mutable value as the default parameter,
## because it is shared across all function calls, so its changed state will persist.
## In this case, though, I want that behavior.
def E_opaque( Y, scratches = {} ):
    return -dot( Y, Y )

def grad_E_opaque( Y, out, scratches = {} ):
    multiply( -2, Y, out )

def E_spatial_static( Y, Ytarget, scratches = {} ):
    if 'Y' not in scratches: scratches['Y'] = Y.copy()
    scratch = scratches['Y']
    
    subtract( Y, Ytarget, scratch )
    return dot( scratch, scratch )

def grad_E_spatial_static( Y, Ytarget, out, scratches = {} ):
    subtract( Y, Ytarget, out )
    out *= 2

def E_spatial_dynamic( Y, LTL, scratches = {} ):
    ## I don't see how to specify the output memory
    return dot( Y, LTL.dot( Y ) )

def grad_E_spatial_dynamic( Y, LTL, out, scratches = {} ):
    ## I don't see how to specify the output memory
    out[:] = LTL.dot( Y )
    out *= 2

def E_polynomial_pieces( Y, C, P, scratches = {} ):
    '''
    Y is a #pix-by-#layers flattened array
    C is a (#layers+1)-by-#channels not-flattened array (the 0-th layer is the background color)
    P is a #pix-by-#channels not-flattened array
    '''
    
    ### Reshape Y the way we want it.
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )
    
    
    ## Allocate scratch space
    if 'F' not in scratches:
        scratches['F'] = empty( P.shape, dtype = Y.dtype )
    F = scratches['F']
    
    if 'M' not in scratches:
        ## We want the non-flattened Y's shape.
        assert len( Y.shape ) > 1
        scratches['M'] = empty( Y.shape, dtype = Y.dtype )
    M = scratches['M']
    
    if 'D' not in scratches:
        scratches['D'] = empty( ( C.shape[0]-1, C.shape[1] ), dtype = Y.dtype )
    D = scratches['D']
    
    if 'DM' not in scratches:
        scratches['DM'] = empty( ( P.shape[0], D.shape[0], D.shape[1] ), dtype = Y.dtype )
    DM = scratches['DM']
    
    if 'energy_presquared' not in scratches:
        scratches['energy_presquared'] = empty( F.shape, dtype = Y.dtype )
    energy_presquared = scratches['energy_presquared']
    
    
    ## Compute F
    subtract( C[newaxis,-1,:], P, F )
    
    ## Compute M
    cumprod( Y[:,::-1], axis = 1, out = M )
    M = M[:,::-1]
    
    ## Compute D
    subtract( C[:-1,:], C[1:,:], D )
    
    ## Finish the computation
    multiply( D[newaxis,...], M[...,newaxis], DM )
    DM.sum( 1, out = energy_presquared )
    energy_presquared += F

def E_polynomial( Y, C, P, scratches = {} ):
    E_polynomial_pieces( Y, C, P, scratches )
    
    energy_presquared = scratches['energy_presquared']
    
    square( energy_presquared, energy_presquared )
    return energy_presquared.sum()

def gradY_E_polynomial( Y, C, P, out, scratches = {} ):
    E_polynomial_pieces( Y, C, P, scratches )
    
    ### Reshape Y the way we want it.
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )
    
    energy_presquared = scratches['energy_presquared']
    D = scratches['D']
    M = scratches['M']
    DM = scratches['DM']
    
    if 'Mi' not in scratches:
        scratches['Mi'] = empty( DM.shape, dtype = Y.dtype )
    Mi = scratches['Mi']
    assert Mi.shape[1] == Y.shape[1]
    
    if 'Yli' not in scratches:
        scratches['Yli'] = empty( Y.shape[0], dtype = Y.dtype )
    Yli = scratches['Yli']
    
    for li in range( Y.shape[1] ):
        Yli[:] = Y[:,li]
        Y[:,li] = 1.
        ## UPDATE: I cannot use cumprod() when aliasing
        ## the input and output parameters and one is the reverse of the other.
        cumprod( Y[:,::-1], axis = 1, out = M )
        Y[:,li] = Yli
        Mr = M[:,::-1]
        Mr[:,li+1:] = 0.
        
        multiply( D[newaxis,...], Mr[...,newaxis], DM )
        DM.sum( 1, out = Mi[:,li,:] )
    
    multiply( energy_presquared[:,newaxis,:], Mi, Mi )
    out.shape = Y.shape
    Mi.sum( 2, out = out )
    out *= 2.
    out.shape = ( prod( Y.shape ), )

def gen_energy_and_gradient( img, layer_colors, weights, img_spatial_static_target = None, scratches = None ):
    '''
    Given a rows-by-cols-by-#channels 'img', where channels are the 3 color channels,
    and (#layers+1)-by-#channels 'layer_colors' (the 0-th color is the background color),
    and a dictionary of floating-point or None weights { w_spatial, w_opacity },
    and an optional parameter 'img_spatial_static_target' which are the target values for 'w_spatial_static' (if not flattened, it will be),
    and an optional parameter 'scratches' which should be a dictionary that will be used to store scratch space between calls to this function (use only *if* arguments are the same size),
    returns a tuple of functions:
        ( e, g )
        where e( Y ) computes the scalar energy of a flattened rows-by-cols-by-#layers array of (1-alpha) values,
        and g( Y ) computes the gradient of e.
    '''
    
    img = asfarray( img )
    layer_colors = asfarray( layer_colors )
    
    assert len( img.shape ) == 3
    assert len( layer_colors.shape ) == 2
    assert img.shape[2] == layer_colors.shape[1]
    
    
    # pprint( weights )
    assert set( weights.keys() ).issubset( set([ 'w_polynomial', 'w_opaque', 'w_spatial_static', 'w_spatial_dynamic' ]) )
    
    C = layer_colors
    P = img.reshape( -1, img.shape[2] )
    
    num_layers = C.shape[0]-1
    Ylen = P.shape[0] * num_layers
    
    if 'w_spatial_static' in weights:
        assert img_spatial_static_target is not None
        Yspatial_static_target = img_spatial_static_target.ravel()
    
    if 'w_spatial_dynamic' in weights:
        # print 'Preparing a Laplacian matrix for E_spatial_dynamic...'
        
        # print '    Generating L...'
        LTL = fast_energy_laplacian.gen_grid_laplacian( img.shape[0], img.shape[1] )
        # print '    Computing L.T*L...'
        # LTL = LTL.T * LTL
        # print '    Replicating L.T*L for all layers...'
        ## Now repeat LTL #layers times.
        ## Because the layer values are the innermost dimension,
        ## every entry (i,j, val) in LTL should be repeated
        ## (i*#layers + k, j*#layers + k, val) for k in range(#layers).
        LTL = LTL.tocoo()
        ## Store the shape. It's a good habit, because there may not be a nonzero
        ## element in the last row and column.
        shape = LTL.shape
        
        ## There is a "fastest" version below.
        '''
        rows = zeros( LTL.nnz * num_layers, dtype = int )
        cols = zeros( LTL.nnz * num_layers, dtype = int )
        vals = zeros( LTL.nnz * num_layers )
        count = 0
        ks = arange( num_layers )
        for r, c, val in zip( LTL.row, LTL.col, LTL.data ):
            ## Slow
            #for k in range( num_layers ):
            #    rows.append( r*num_layers + k )
            #    cols.append( c*num_layers + k )
            #    vals.append( val )
            
            ## Faster
            rows[ count : count + num_layers ] = r*num_layers + ks
            cols[ count : count + num_layers ] = c*num_layers + ks
            vals[ count : count + num_layers ] = val
            count += num_layers
            
        assert count == LTL.nnz * num_layers
        '''
        
        ## Fastest
        ks = arange( num_layers )
        rows = ( repeat( asarray( LTL.row ).reshape( LTL.nnz, 1 ) * num_layers, num_layers, 1 ) + ks ).ravel()
        cols = ( repeat( asarray( LTL.col ).reshape( LTL.nnz, 1 ) * num_layers, num_layers, 1 ) + ks ).ravel()
        vals = ( repeat( asarray( LTL.data ).reshape( LTL.nnz, 1 ), num_layers, 1 ) ).ravel()
        
        LTL = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( shape[0]*num_layers, shape[1]*num_layers ) ).tocsr()
        # print '...Finished.'
    
    if scratches is None:
        scratches = {}
    
    def e( Y ):
        e = 0.
        
        if 'w_polynomial' in weights:
            e += weights['w_polynomial'] * E_polynomial( Y, C, P, scratches )
        
        if 'w_opaque' in weights:
            e += weights['w_opaque'] * E_opaque( Y, scratches )
        
        if 'w_spatial_static' in weights:
            e += weights['w_spatial_static'] * E_spatial_static( Y, Yspatial_static_target, scratches )
        
        if 'w_spatial_dynamic' in weights:
            e += weights['w_spatial_dynamic'] * E_spatial_dynamic( Y, LTL, scratches )
        
        # print 'Y:', Y
        # print 'e:', e
        
        return e
    
    ## Preallocate this memory
    gradient_space = [ zeros( Ylen ), zeros( Ylen ) ]
    # total_gradient = zeros( Ylen )
    # gradient_term = zeros( Ylen )
    
    def g( Y ):
        total_gradient = gradient_space[0]
        gradient_term = gradient_space[1]
        
        total_gradient[:] = 0.
        
        if 'w_polynomial' in weights:
            gradY_E_polynomial( Y, C, P, gradient_term, scratches )
            gradient_term *= weights['w_polynomial']
            total_gradient += gradient_term
        
        if 'w_opaque' in weights:
            grad_E_opaque( Y, gradient_term, scratches )
            gradient_term *= weights['w_opaque']
            total_gradient += gradient_term
        
        if 'w_spatial_static' in weights:
            grad_E_spatial_static( Y, Yspatial_static_target, gradient_term, scratches )
            gradient_term *= weights['w_spatial_static']
            total_gradient += gradient_term
        
        if 'w_spatial_dynamic' in weights:
            grad_E_spatial_dynamic( Y, LTL, gradient_term, scratches )
            gradient_term *= weights['w_spatial_dynamic']
            total_gradient += gradient_term
        
        # print 'Y:', Y
        # print 'total_gradient:', total_gradient
        
        return total_gradient
    
    return e, g

def composite_layers( layers ):
    layers = asfarray( layers )
    # print("layers.shape",layers.shape)
    
    ## Start with opaque white.
    out = 255*ones( layers[0].shape)[:,:,:3]# with no alpha
    for layer in layers:
        change = layer[:,:,3:]/255*( layer[:,:,:3] - out )
        out += change
        # print("this layer:", change[0,1,:])
        # print("out", out[0,1,:])
    
    return out

def optimize( arr, colors, Y0, weights, img_spatial_static_target = None, scratches = None, saver = None ):
    '''
    Given a rows-by-cols-by-#channels array 'arr', where channels are the 3 color channels,
    and (#layers+1)-by-#channels 'colors' (the 0-th color is the background color),
    and rows-by-cols-by-#layers array 'Y0' of initial (1-alpha) values for each pixel (flattened or not),
    and a dictionary of floating-point or None weights { w_polynomial, w_opacity, w_spatial_dynamic, w_spatial_static },
    and an optional parameter 'img_spatial_static_target' which are the target values for 'w_spatial_static' (if not flattened, it will be),
    and an optional parameter 'scratches' which should be a dictionary that will be used to store scratch space between calls to this function (use only *if* arguments are the same size),
    and an optional parameter 'saver' which will be called after every iteration with the current state of Y.
    returns a rows-by-cols-#layers array of optimized Y values, which are (1-alpha).
    '''
    
    

    start = time.clock()

    Y0 = Y0.ravel()
    
    Ylen = len( Y0 )
    
    e, g = gen_energy_and_gradient( arr, colors, weights, img_spatial_static_target = img_spatial_static_target, scratches = scratches )
    
    bounds = zeros( ( Ylen, 2 ) )
    bounds[:,1] = 1.
    
    ## Save the result-in-progress in case the users presses control-C.
    ## [number of iterations, last Y]
    Ysofar = [0,None]
    def callback( xk ):
        Ysofar[0] += 1
        ## Make a copy
        xk = array( xk )
        Ysofar[1] = xk
        
        if saver is not None: saver( xk )
    
    # print 'Optimizing...'
    # start = time.clock()
    
    try:
        ## WOW! TNC does a really bad job on our problem.
        # opt_result = scipy.optimize.minimize( e, Y0, method = 'TNC', jac = g, bounds = bounds )
        ## I did an experiment with the 'tol' parameter.
        ## I checked in the callback for a max/total absolute difference less than 1./255.
        ## Passing tol directly doesn't work, because the solver we are using (L-BFGS-B)
        ## normalizes it by the maximum function value, whereas we want an
        ## absolute stopping criteria.
        ## Max difference led to stopping with visible artifacts.
        ## Total absolute difference terminated on the very iteration that L-BFGS-B did
        ## anyways.
        opt_result = scipy.optimize.minimize( e, Y0, jac = g, bounds = bounds, callback = callback )
    
    except KeyboardInterrupt:
        ## If the user 
        print ('KeyboardInterrupt after %d iterations!' % Ysofar[0])
        Y = Ysofar[1]
        ## Y will be None if we didn't make it through 1 iteration before a KeyboardInterrupt.
        if Y is None:
            Y = -31337*ones( ( arr.shape[0], arr.shape[1], len( colors )-1 ) )
    
    else:
        # print opt_result
        Y = opt_result.x
    
    # duration = time.clock() - start
    # print '...Finished optimizing in %.3f seconds.' % duration
    
    end = time.clock()
    print ('Optimize an image of size ', Y.shape, ' took ', (end-start), ' seconds.')

    Y = Y.reshape( arr.shape[0], arr.shape[1], len( colors )-1 )
    return Y

def run_one( color_list, orderpath, OrderPath, colorpath, outprefix, weightspath = None, save_every = None, solve_smaller_factor = None, too_small = None ):
    '''
    Given a path `imgpath` to an image,
    a path `colorpath` to a JSON file containing an array of RGB triplets of layer colors (the 0-th color is the background color),
    a prefix `outprefix` to use for saving files,
    an optional path `weightspath` to a JSON file containing a dictionary of weight values,
    an optional positive number `save_every` which specifies how often to save progress,
    an optional positive integer `solve_smaller_factor` which, if specified,
    will first solve on a smaller image whose dimensions are `1/solve_smaller_factor` the full size image,
    and an optional positive integer `too_small` which, if specified, determines
    the limit of the `solve_smaller_factor` recursion as the minimum image size (width or height),
    runs optimize() on it and saves the output to e.g. `outprefix + "-layer01.png"`.
    '''
    
    import json, os
    from PIL import Image
    import time
    
    # arr = asfarray( Image.open( imgpath ).convert( 'RGB' ) )
    arr = color_list
    print("the shape of array is",arr.shape)
    arr_backup=arr.copy()
    arr = arr/255.0
    order=loadtxt(orderpath).astype(uint8)
    print (order)
    colors = asfarray(json.load(open(colorpath))['vs'])
    colors_backup=colors.copy()
    # print (colors)
    colors=colors[order,:]/255.0
    # print (colors*255.0)
    
    assert solve_smaller_factor is None or int( solve_smaller_factor ) == solve_smaller_factor
    
    if save_every is None:
        save_every = 10.
    
    if too_small is None:
        too_small = 5
    
    # arr = arr[:1,:1,:]
    # colors = colors[:3]
    
    kSaveEverySeconds = save_every

    print ("Will Save temporary results every "+str(kSaveEverySeconds)+" seconds!")
    print( """If you do not want to save temporary results, you can increase value to look like "--save-every 10000" in command line""")

    ## [ number of iterations, time of last save, arr.shape ]
    last_save = [ None, None, None ]
    def reset_saver( arr_shape ):
        last_save[0] = 0
        last_save[1] = time.clock()
        last_save[2] = arr_shape
    def saver( xk ):
        arr_shape = last_save[2]
        
        last_save[0] += 1
        now = time.clock()
        ## Save every 10 seconds!
        if now - last_save[1] > kSaveEverySeconds:
            print ('Iteration', last_save[0])
            # save_results( xk, colors,"none")
            ## Get the time again instead of using 'now', because that doesn't take into
            ## account the time to actually save the images, which is a lot for large images.
            last_save[1] = time.clock()
    
    Ylen = arr.shape[0]*arr.shape[1]*( len(colors) - 1 )
    
    # Y0 = random.random( Ylen )
    # Y0 = zeros( Ylen ) + 0.0001
    Y0 = .5*ones( Ylen )
    # Y0 = ones( Ylen )
    
    static = None
    if weightspath is not None:
        weights = json.load( open( weightspath ) )
    else:
        weights = { 'w_polynomial': 3.0, 'w_opaque': 500., 'w_spatial_dynamic': 5000. }
        # weights = { 'w_polynomial': 1., 'w_opaque': 100. }
        # weights = { 'w_opaque': 100. }
        # weights = { 'w_spatial_static': 100. }
        # static = 0.75 * ones( Ylen )
        # weights = { 'w_spatial_dynamic': 100. }
        # weights = { 'w_spatial_dynamic': 100., 'w_opaque': 100. }

    num_layers=len(colors)-1
    ### adjust the weights:
    if 'w_polynomial' in weights:
        weights['w_polynomial'] *= 50000.0 #### old one is 255*255
        weights['w_polynomial'] /= arr.shape[2]
    
    if 'w_opaque' in weights:
        weights['w_opaque'] /= num_layers
    
    if 'w_spatial_static' in weights:
        weights['w_spatial_static'] /= num_layers
    
    if 'w_spatial_dynamic' in weights:
        weights['w_spatial_dynamic'] /= num_layers
     

    
    if solve_smaller_factor is not None:
        assert solve_smaller_factor > 1
        
        def optimize_smaller( solve_smaller_factor, large_arr, large_Y0, large_img_spatial_static_target ):
            ## Terminate recursion if the image is too small.
            if large_arr.shape[0]//solve_smaller_factor < too_small or large_arr.shape[1]//solve_smaller_factor < too_small:
                return large_Y0
            
            ## small_arr = downsample( large_arr )
            small_arr = large_arr[::solve_smaller_factor,::solve_smaller_factor]
            ## small_Y0 = downsample( large_Y0 )
            small_Y0 = large_Y0.reshape( large_arr.shape[0], large_arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
            ## small_img_spatial_static_target = downsample( large_img_spatial_static_target )
            small_img_spatial_static_target = None
            if large_img_spatial_static_target is not None:
                small_img_spatial_static_target = large_img_spatial_static_target.reshape( arr.shape[0], arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
            
            ## get an improved Y by recursively shrinking
            small_Y1 = optimize_smaller( solve_smaller_factor, small_arr, small_Y0, small_img_spatial_static_target )
            
            ## solve on the downsampled problem
            print ('==> Optimizing on a smaller image:', small_arr.shape, 'instead of', large_arr.shape)
            reset_saver( small_arr.shape )
            small_Y = optimize( small_arr, colors, small_Y1, weights, img_spatial_static_target = small_img_spatial_static_target, saver = saver )
            
            ## save the intermediate solution.
            saver( small_Y )
            
            ## large_Y1 = upsample( small_Y )
            ### 1 Make a copy
            large_Y1 = array( large_Y0 ).reshape( large_arr.shape[0], large_arr.shape[1], -1 )
            ### 2 Fill in as much as will fit using numpy.repeat()
            small_Y = small_Y.reshape( small_arr.shape[0], small_arr.shape[1], -1 )
            small_Y_upsampled = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
            large_Y1[:,:] = small_Y_upsampled[ :large_Y1.shape[0], :large_Y1.shape[1] ]
            # large_Y1[ :small_Y.shape[0]*solve_smaller_factor, :small_Y.shape[1]*solve_smaller_factor ] = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
            ### 3 The right and bottom edges may have been missed due to rounding
            # large_Y1[ small_Y.shape[0]*solve_smaller_factor:, : ] = large_Y1[ small_Y.shape[0]*solve_smaller_factor - 1 : small_Y.shape[0]*solve_smaller_factor, : ]
            # large_Y1[ :, small_Y.shape[1]*solve_smaller_factor: ] = large_Y1[ :, small_Y.shape[1]*solve_smaller_factor - 1 : small_Y.shape[1]*solve_smaller_factor ]
            
            return large_Y1.ravel()
        
        Y0 = optimize_smaller( solve_smaller_factor, arr, Y0, static )
    
    reset_saver( arr.shape )
    Y = optimize( arr, colors, Y0, weights, img_spatial_static_target = static, saver = saver )
    
    composite_img=save_results( Y, colors , 'reconstruction_direct_')
    img_diff=composite_img-arr_backup
    RMSE=sqrt(square(img_diff).sum()/(composite_img.shape[0]*composite_img.shape[1]))
    
    # print ('img_shape is: ', img_diff.shape)
    # print ('max diff: ', sqrt(square(img_diff).sum(axis=2)).max())
    # print ('median diff', median(sqrt(square(img_diff).sum(axis=2))))
    # print ('RMSE: ', RMSE)


    ##### save alphas as barycentric coordinates
    alphas=1. - Y.reshape((arr.shape[0]*arr.shape[1], -1 ))
    # print("alphas:", alphas[:6])
    extend_alphas=ones((alphas.shape[0],alphas.shape[1]+1))
    extend_alphas[:,1:]=alphas
    # print("extend_alphas from deposition:",extend_alphas)

    # savetxt(outprefix +'-'+str(len(colors))+'-PD_layers-'+os.path.splitext(OrderPath)[0]+"-opacities.txt",extend_alphas)

    #### first columns of extend_alphas are all 1.0
    barycentric_weights=covnert_from_alphas_to_barycentricweights(extend_alphas)

    origin_order_barycentric_weights=ones(barycentric_weights.shape)
    #### to make the weights order is same as orignal input vertex order
    origin_order_barycentric_weights[:,order]=barycentric_weights

    # test_weights_diff1=origin_order_barycentric_weights-barycentric_weights
    # test_weights_diff2=barycentric_weights-barycentric_weights
    # print len(test_weights_diff1[test_weights_diff1==0])
    # print len(test_weights_diff2[test_weights_diff2==0])


    ####assert
    temp=sum(origin_order_barycentric_weights.reshape((origin_order_barycentric_weights.shape[0],origin_order_barycentric_weights.shape[1],1))*colors_backup, axis=1)
    diff=temp-arr_backup.reshape((-1,3))
    # assert(abs(diff).max()<0.5)
    print (abs(diff).max())
    print (diff.shape[0])
    print (sqrt(square(diff).sum()/diff.shape[0]))


    origin_order_barycentric_weights=origin_order_barycentric_weights.reshape((arr.shape[0],arr.shape[1],-1))



    import json
    output_all_weights_filename=outprefix+'-'+str(len(colors))+"-PD_mixing-weights.js"
    with open(output_all_weights_filename,'wb') as myfile:
        x = {'weights': origin_order_barycentric_weights.tolist()}
        # json.dump(x, myfile)
        s = json.dumps(x)
        myfile.write(s.encode())
    
    for i in range(origin_order_barycentric_weights.shape[-1]):
        output_all_weights_map_filename=outprefix+'-'+str(len(colors))+"-PD_mixing-weights_map-%02d.png" % i
        Image.fromarray((origin_order_barycentric_weights[:,:,i]*255).round().clip(0,255).astype(uint8)).save(output_all_weights_map_filename)
    return Y, origin_order_barycentric_weights
    

import numpy as np
def covnert_from_alphas_to_barycentricweights(alphas,epsilon=0.0):
    
#### first column of alphas should be all 1.0 (canvas is set to be opaque)
    # def get_weight_from_alpha(alpha,epsilon=0.0):
    #     weight=np.ones(len(alpha))
    #     # print("weight",weight)
    #     # print("alpha",alpha)
    #     for i in range(len(weight)-1):
    #         temp1=1.0
    #         temp2=1.0
    #         for j in range(i,len(weight)):
    #             temp1*=(1.0-alpha[j])
    #         for j in range(i+1,len(weight)):
    #             temp2*=(1.0-alpha[j])
    #         weight[i]=temp2-temp1
    #     weight[-1]=alpha[-1]
    #     return weight
    def get_weight_from_alpha(alpha,epsilon=0.0):
        weight=np.ones(len(alpha))
        # print("weight",weight)
        # print("alpha",alpha)
        for i in range(len(weight)-1):
            temp2=1.0
            for j in range(i+1,len(weight)):
                temp3 = (1.0-alpha[j])
                if temp3 == 0:
                    temp3 = 0.0001
                temp2*=(temp3)
            temp1 = alpha[i]
            weight[i]=temp2*temp1
        weight[-1]=alpha[-1]
        return weight

    # print("alphas.shape", alphas.shape)
    weights=np.zeros(alphas.shape)
    for ind in range(len(weights)):
        alpha=alphas[ind]
        weights[ind]=get_weight_from_alpha(alpha)
    return weights

def get_alpha_from_weight(weight, epsilon=0.0):
    alpha = np.ones(len(weight))
    alpha[-1] = weight[-1]
    for i in reversed(range(len(weight)-1)):
        temp2 = 1.0
        for j in reversed(range(i+1,len(weight))):
            temp1 = 1-alpha[j]
            if temp1 == 0:
                temp1 = 0.0001
            temp2 *= temp1
        alpha[i] = weight[i]/temp2
    return alpha

def covnert_from_barycentricweights_to_alphas(oobw,epsilon=0.0):
    alphas=np.zeros(oobw.shape)
    for ind in range(len(alphas)):
        bw=oobw[ind]
        for inx in range(alphas.shape[1]):
            weight = bw[inx]
            alphas[ind][inx]=get_alpha_from_weight(weight, epsilon=0.0)
    return alphas


def save_results( Y, parlette_pigments, note):
    print(Y.shape)
    (h,w,l) = Y.shape
    img_shape = [h,w,3] 
    alphas = 1. - Y.reshape( img_shape[0], img_shape[1], -1 )
    layers = []
    for li, color in enumerate( parlette_pigments ):  ### colors are now in range[0.0,1.0] not [0,255]
        layer = ones( ( img_shape[0], img_shape[1], 4 ), dtype = uint8 )
        layer[:,:,:3] = asfarray(color*255.0).round().clip( 0,255 ).astype( uint8 )
        layer[:,:,3] = 255 if ( li == 0 ) else (alphas[:,:,li-1]*255.).round().clip( 0,255 ).astype( uint8 )
        layers.append( layer )

        outpath = '-opacities_map-%02d.png' % li
        Image.fromarray( layer[:,:,3] ).save( outpath )
    layers = np.array(layers)
    
    composited = composite_layers( layers )
    composited = composited.round().clip( 0, 255 ).astype( uint8 )
    # outpath = outprefix+'-'+str(len(colors))+'-PD_layers-'+os.path.splitext(OrderPath)[0]+ '-reconstructed.png'
    outpath =note+'reconstructed.png'
    Image.fromarray( composited ).save( outpath )
    return composited

# if __name__ == '__main__':
    
#     def usage():
#         print ( sys.stderr, "Usage:", sys.argv[0], "path/to/image path/to/layer_color_order path/to/layer_color_list.js path/to/output [--weights /path/to/weights.js] [--save-every save_every_N_seconds N] [--solve-smaller-factor F] [--too-small T]")
#         print (sys.stderr, "NOTE: The 0-th element of layer_color_list is the background color.")
#         print ( sys.stderr, 'NOTE: Files will be saved to "path/to/output-reconstructed.png" and "path/to/output-layer01.png"')
#         sys.exit(-1)
    
#     args = list( sys.argv[1:] )
    
#     try:
        
#         weightspath = None
#         try:
#             index = args[:-1].index( '--weights' )
#             weightspath = args[ index+1 ]
#             del args[ index : index+2 ]
#         except ValueError: pass
        
#         save_every = None
#         try:
#             index = args[:-1].index( '--save-every' )
#             save_every = int( args[ index+1 ] )
#             del args[ index : index+2 ]
#         except ValueError: pass
        
#         solve_smaller_factor = None
#         try:
#             index = args[:-1].index( '--solve-smaller-factor' )
#             solve_smaller_factor = int( args[ index+1 ] )
#             del args[ index : index+2 ]
#         except ValueError: pass
        
#         too_small = None
#         try:
#             index = args[:-1].index( '--too-small' )
#             too_small = int( args[ index+1 ] )
#             del args[ index : index+2 ]
#         except ValueError: pass
    
#     except Exception:
#         usage()
    
#     if len( args ) != 4: usage()

#     global OrderPath
#     foldername, image_path, OrderPath, color_path = args

#     current_folder="."+foldername+"/" 

#     # output_prefix=current_folder+"Tan2016_PD_results/"
#     output_prefix=current_folder
#     make_sure_path_exists(output_prefix)

#     output_prefix+=os.path.splitext(image_path)[0]
#     color_list = array([[[155,27,3],[243,133,45],[12,53,89]]])



#     start=time.clock()
#     result = run_one( color_list, current_folder+OrderPath, OrderPath,current_folder+"Tan2016_PD_results/"+color_path, output_prefix, weightspath = current_folder+weightspath, save_every = save_every, solve_smaller_factor = solve_smaller_factor, too_small = too_small )
#     end=time.clock()
#     print ('time: ', end-start)
#     print('result:', result)

