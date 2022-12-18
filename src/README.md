# Pigmento: Pigment-Based Image Analysis and Editing


This code implements the pipeline described in the IEEE TVCG 2018 paper ["Pigmento: Pigment-Based Image Analysis and Editing"](https://cragl.cs.gmu.edu/pigmento/) Jianchao Tan, Stephen DiVerdi, Jingwan Lu and Yotam Gingold.

* The input image should be a good size, like 600*600. If it is large image, it will be slow for weights map extraction or layer map extraction.
* The flower image is from [Nel Jansen](https://nelseverydaypainting.blogspot.com/2011/04/dogwood-in-hood.html#links).

#### Dependencies:
 * Python 2.7
 * Numpy
 * Scipy
 * Pillow
 * numba
 * matplotlib
 * autograd
 * joblib
 * scikit-image
 * scikit-learn
 * cvxopt (with 'glpk' solver option)
 * opencv3 (brew install opencv3)


#### Main code files:
* step1_ANLS_with_autograd.py
* Solve_KM_mixing_model_fixed_KS_with_autograd.py
* Solve_KM_layer_model_fixed_KS_with_autograd.py
* fast_energy_RGB_lap_adjusted_weights.py
* Editing_GUI.py



#### Commands (run 1 first, then run 2 or 3, 4 is for comparison, 5 is for GUI): 

##### 1. Extract KM primary pigments: 
User can give number of pigments, for example, "6" in the command line below.
```sh
	$ cd new_pipeline_executable

	$ python step1_ANLS_with_autograd.py wheatfield-crop-steve.png Existing_KS_parameter_KS.txt 2 None wheatfield-crop-steve-sampled_pixels-400 0 6 10.0 0.0 0.0 0.001 0.001 1e-6 /wheatfield-crop None 0 1 1000 400 1 0
```


##### 2. Extract KM mixing weights (All layer results in our Pigmento paper are this option!):
You can use default parameter values in command line directly, only need change example name.

```sh
	$ cd new_pipeline_executable/wheatfield-crop

	$ python ../Solve_KM_mixing_model_fixed_KS_with_autograd.py wheatfield-crop-steve.png  primary_pigments_KS-6.txt  None wheatfield-crop-steve-primary_pigments_color_vertex-6-KM_weights-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes 10.0 0.1 0 1.0 0.0 blf Yes
```



##### 3. Extract KM layers: 
You need create a layer order file manually: "order1.txt" and put it in /wheatfield-crop folder, since we are using 6 pigments, so "order1.txt" content can be like: 0 1 2 3 4 5 or their permutations.Then you can run below command.
```sh
	$ cd new_pipeline_executable/wheatfield-crop

	$ python ../Solve_KM_layer_model_fixed_KS_with_autograd.py wheatfield-crop-steve.png  primary_pigments_KS-6.txt  None wheatfield-crop-steve-primary_pigments_color_vertex-6-KM_layers-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes-order1 10.0 0.1 0 1.0 0.0 blf Yes order1.txt
```



##### 4. Extract PD layers and weights (Tan 2016) using KM pigments's RGB colors as primary color. It will use same order as KM layers. 
```sh
	$ cd new_pipeline_executable

	$ python fast_energy_RGB_lap_adjusted_weights.py  /wheatfield-crop wheatfield-crop-steve.png order1.txt primary_pigments_color_vertex-6.js  --weights weights-poly3-opaque400-dynamic40000.js  --solve-smaller-factor 2 --save-every 50
```



##### 5. GUI code. Above commands will generate a "Application_Files" folder in the "wheatfield-crop" folder, which will contain all needed files for GUI. 
```sh
	$ cd new_pipeline_executable

	$ python Editing_GUI.py
```





# For whole pipeline in Tan2016

#### 1. Extract palettes:
All results will saved in a created folder "Tan2016_PD_results", "/wheatfield-crop" in below command is example folder name, and "6" is palette size that user choose.

```sh
	$ cd new_pipeline_executable

	$ python Extract_PD_palettes.py wheatfield-crop-steve.png 6 /wheatfield-crop
```


#### 2. Extract layers. 
Like before, you need a different "order1.txt" file for different examples, and same "weights-poly3-opaque400-dynamic40000.js" for all examples. You can just simply set "order1.txt" content as 0 1 2 3 4 5 in this example.  All results will saved in a created folder "Tan2016_PD_results". Actually, below command format is same as before, I only change the code filename to "Extract_PD_layers.py" and change input palette filename to "wheatfield-crop-steve-6-PD_palettes.js". 

```sh
	$ cd new_pipeline_executable

	$ python Extract_PD_layers.py  /wheatfield-crop wheatfield-crop-steve.png order1.txt wheatfield-crop-steve-6-PD_palettes.js  --weights weights-poly3-opaque400-dynamic40000.js  --solve-smaller-factor 2 --save-every 50
```


#### For Tan2016 global recoloring GUI:

##### 1. Go to this link: [Tan2016 PD recoloring GUI](https://yig.github.io/image-rgb-in-3D/)
##### 2. Drag an image into the browser window. For example: "wheatfield-crop-steve.png". Rotate by dragging with the mouse.
##### 3. Drag a convex hull JSON file into the browser window. For example: "wheatfield-crop-steve-6-PD_palettes.js"
##### 4. Drag a weights JSON file into the browser window. For example: "wheatfield-crop-steve-6-PD_mixing-weights.js"
##### 5. Recolor by dragging vertices of the convex hull. Recoloring results are shown in real-time. Click the recolored image to save it to disk, Or you can also click "Save_Everything" in browser to save changed everything, including changed vertices. 

##### Tip: Convex hull vertices are dragged parallel to the view plane. It is easy to drag the vertex outside the RGB cube (imaginary color). This works, but may not be desired. It may be necessary to interleave vertex dragging with camera rotation to move a vertex to a desired location.
