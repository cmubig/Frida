from Extract_PD_layers import *
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
import numpy as np
import json
import mixbox
from multimix import *

def compute_weight(color):
    foldername = '/live-photo'
    color_list = color
    OrderPath = 'order2palette.txt'
    pigmnet_path = '2-PD_palettes.js'

    weightspath = 'weights-poly3-opaque400-dynamic40000.js'
    save_every = 50
    solve_smaller_factor = 2
    too_small = None

    current_folder="."+foldername+"/" 

    # output_prefix=current_folder+"Tan2016_PD_results/"
    output_prefix=current_folder
    make_sure_path_exists(output_prefix)

    start=time.clock()
    Y, oobw = run_one( color_list, current_folder+OrderPath, OrderPath, current_folder+"Tan2016_PD_results/"+pigmnet_path, output_prefix, weightspath = current_folder+weightspath, save_every = save_every, solve_smaller_factor = solve_smaller_factor, too_small = too_small )
    end=time.clock()


    # print('OOBW:', oobw)



    return Y, oobw
    # print ('time: ', end-start)
    # print('result:', result)

def reconstruct_img(Y):
    current_folder="./live-photo/"
    OrderPath = 'order2palette.txt'
    pigmnet_path = '2-PD_palettes.js'
    pigments = np.asfarray(json.load(open(current_folder+"Tan2016_PD_results/"+pigmnet_path))['vs'])
    order=loadtxt(current_folder+OrderPath).astype(uint8)
    pigments=pigments[order,:]/255.0
    composite_img=save_results( Y, pigments,"none")
    # print("reconstruction image:", composite_img.shape)
    return composite_img

def reconstruct_img_from_weights_mixbox(oobw):
    current_folder="./live-photo/"
    pigmnet_path = '2-PD_palettes.js'
    pigments = np.asfarray(json.load(open(current_folder+"Tan2016_PD_results/"+pigmnet_path))['vs'])
    h,w, c = oobw.shape
    reconstruct_img = np.ones((h,w,3))
    for i in range(reconstruct_img.shape[0]):
        for j in range(reconstruct_img.shape[1]):
            pixel_rgb = reconstruct_single_color(pigments, oobw[i][j])
            pixel_rgb = np.array(pixel_rgb)
            reconstruct_img[i,j,:] = pixel_rgb 
    # print("reconstruction image:", reconstruct_img[0,:,:])
    reconstruct_img = reconstruct_img.round().clip( 0, 255 ).astype(uint8)
    outpath ='test_reconstructed.png'
    Image.fromarray( reconstruct_img ).save( 'test_reconstructed.jpg' )
    return reconstruct_img

def reconstruct_img_from_weights(oobw):
    current_folder="./live-photo/"
    OrderPath = 'order2palette.txt'
    pigmnet_path = '2-PD_palettes.js'
    pigments = np.asfarray(json.load(open(current_folder+"Tan2016_PD_results/"+pigmnet_path))['vs'])
    order=loadtxt(current_folder+OrderPath).astype(uint8)
    pigments=pigments[order,:]/255.0
    h,w, c = oobw.shape

    extend_alphas = covnert_from_barycentricweights_to_alphas(oobw,epsilon=0.0)
    alphas = extend_alphas[:,:,1:]
    # print('alphas from reconstruction',alphas[0])
    Y = 1 - alphas.reshape((h,w,c-1))
    # print("Y from reconstruction:",Y[0])
    composite_img=save_results( Y, pigments,"barycentric_" )
    # composite_img = composite_img.round().clip( 0, 255 ).astype(uint8)
    return Y, composite_img






# color = np.array([[[45,15,67],[54,13,4],[243,133,45],[155,27,3],[23,15,157]]])
# color = np.asfarray(Image.open('swatch48.png').convert('RGB'))
# Y, oobw = compute_weight(color)
# print("Y from deposition", Y[0])


# reconstruct_imge = reconstruct_img_from_weights_mixbox(oobw)
# composite_img = reconstruct_img_from_weights(oobw)
# print("color",color[0,:,:])
# print("reconstruct_imge from mixbox",reconstruct_imge[0,:,:] )
# print("composite_img",composite_img[0,:,:])
