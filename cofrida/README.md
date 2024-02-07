# CoFRIDA
[Peter Schaldenbrand](https://pschaldenbrand.github.io/#about.html), [Gaurav Parmar](https://gauravparmar.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Jim McCann](http://www.cs.cmu.edu/~jmccann/), and [Jean Oh](https://www.cs.cmu.edu/~./jeanoh/)

The Robotics Institute, Carnegie Mellon University


### System Requirements

We recommend running FRIDA on a machine with Python 3.8 and Ubuntu (we use 20.04). FRIDA's core functionality uses CUDA, so it is recommended to have an NVIDIA GPU with 8+Gb vRAM. Because CoFRIDA uses Stable Diffusion, it is recommended to have 12+Gb for running and 16+Gb vRam for training CoFRIDA.

### Code Installation

```
git clone https://github.com/pschaldenbrand/Frida.git

# Install CUDA

# We use Python 3.8
cd Frida
pip3 install --r requirements.txt

# For training CoFRIDA, you'll need additional installation steps
cd Frida/src
pip3 install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
git clone https://github.com/jmhessel/clipscore.git
```

### Create Data for Training CoFRIDA

See `Frida/cofrida/create_copaint_data.sh` for an example on command-line arguments.

In the data creation phase, paintings/drawings are created in the FRIDA simulation either from images in the CoCo dataset or from Stable Diffusion images generated from the Parti Prompts dataset.  Strokes are selectively removed to form partial paintings/drawings.

```
python3 create_copaint_data.py 
        [--use_cache] Load a pre-trained brush stroke model
        [--cache_dir path] Path to brush model
        [--materials_json path] Path to file describing the materials used by FRIDA
        [--lr_multiplier float] Scale the learning rate for data generation
        [--n_iters int] Number of optimization iterations to generate each training image
        [--max_strokes_added int] Max. number of strokes in full painting/drawing
        [--min_strokes_added int] Min. number of strokes in full painting/drawing
        [--ink] Use just black strokes
        [--output_parent_dir path] Where to save the data
        [--max_images int] Maximum number of training images to create
        [--colors [[r,g,b],]] Specify a specific color palette to use. If None, use any color palette (discretized to --n_colors)
```

#### What images to use as training data

CoFRIDA needs a dataset of image-text pairs to use to create full and partial paintings for training.

To use the CoCo dataset image-text pairs for training: `--cofrida_dataset ChristophSchuhmann/MS_COCO_2017_URL_TEXT`

To generate images using Stable Diffusion conditioned on Parti Prompts Dataset: `--generate_cofrida_training_data --cofrida_dataset nateraw/parti-prompts` 

### Train CoFRIDA Model

See `Frida/cofrida/train_instruct_pix2pix.sh` for an example on command-line arguments.

CoFRIDA fine-tunes a pre-trained Instruct-Pix2Pix model to translate from partial to full drawings/paintings conditioned on a text description.

```
export MODEL_DIR="timbrooks/instruct-pix2pix"
export OUTPUT_DIR="./cofrida_model_ink/"

accelerate launch train_instruct_pix2pix.py 
    [--pretrained_model_name_or_path] Pretrained instruct-pix2pix model to use 
    [--data_dict path] Where to find dictionary describing training data (see --output_parent_dir used with create_copaint_data.py) 
    [--output_dir path] Path to where to save trained models 
    [--resolution int]  
    [--learning_rate float]  
    [--train_batch_size int]
    [--gradient_accumulation_steps int]
    [--gradient_checkpointing]
    [--tracker_project_name string] Name for TensorBoard logs
    [--validation_steps int] After how many steps to log validation cases 
    [--num_train_epochs int] Number of times to go through training data 
    [--validation_image paths] List of paths to images as conditioning for validation cases
    [--validation_prompt strings] List of text prompts for validation cases
    [--use_8bit_adam] 
    [--num_validation_images int] Number of times to run each validation case
    [--seed int] 
    [--logging_dir path] Path to where to save TensorBoard logs
```
#### Monitor Training Logs
```
tensorboard --logdir [--output_dir from train_instruct_pix2pix.py]/logs
```

### Run CoFRIDA w/ Robot

```
cd Frida/src/

python3 codraw.py 
    [--cofrida_model path] Path to trained Instruct-Pix2Pix (see --output_dir used with train_instruct_pix2pix.py)
python3 codraw.py  
        --use_cache 
        --cache_dir caches/cache_6_6_cvpr/ 
        --dont_retrain_stroke_model 
        --robot xarm 
        --brush_length 0.2 
        --ink   
        --lr_multiplier 0.3 
        --num_strokes 120
# Example below
python3 codraw.py --use_cache --cache_dir caches/cache_6_6_cvpr/ --cofrida_model ../cofrida/cofrida_model_ink --dont_retrain_stroke_model --robot xarm --brush_length 0.2 --ink   --lr_multiplier 0.3 --num_strokes 120 --simulate
```


### Test CoFRIDA on the Computer