# FRIDA: A Collaborative Robot Painter with a Differentiable, Real2Sim2Real Simulated Planning Environment

Peter Schaldenbrand, Jean Oh, Jim McCann


# Run in Simulation

```
cd src/
python3 plan.py --simulate [other args see below]
```

# Arguments

# Run with a robot


## Acknowledgements

Thank you to Jia Chen Xu for writing FRIDA's perception code! Thank you to Heera Sekhr and Jesse Ding for their help in the early stages of designing FRIDA's planning algorithms.  Thank you to Vihaan Misra for writing a sketch loss function.


```
# Run robot
rosrun paint paint.py --target /home/frida/Downloads/andy.jpg  --n_colors 9  --max_height 196 --num_strokes 70 --adaptive --use_cache --simulate --n_stroke_models 1 --init_objective l2 --init_objective_data /home/frida/Downloads/georgia.jpg --init_objective_weight 0.2 --objective text --objective_data "A car" --objective_weight 1.0  --objective style --objective_data /home/frida/Downloads/georgia.jpg  --objective_weight 0.1

# Run Sim add the --simulate commandline arg

# Just plan using python3 (Colab)
cd scripts
python3 plan.py [args]
```

# Everything below this is pretty old

```
git clone https://github.com/pschaldenbrand/SawyerPainter.git
cd SawyerPainter/scripts/

# Might need to run the following line, especially if you get a pickle error
dos2unix cache/*

# Run the simulation
python paint.py --use_cache --simulate --cache_dir cache --target frida.jpg

# In another terminal, run this to view progress
tensorboard --logdir SawyerPainter/scripts/painting

# Open browser and navigate to http://localhost:6006/
```

## Options
See options in scripts/options.py for commandline parameters and robot parameters (e.g., location of paint/canvas/water).


```
python paint.py [--target path] [--use_cache] [--simulate] [--cache_dir path] [--n_colors int]
```

- `--target` - Path to image to paint
- `--use_cache` - Use cached stroke library and robot parameters (use this for simulation)
- `--simulate` - Use simulated painting environment
- `--n_colors` - The number of discrete paint colors to use
- `--cache_dir` - Location of cached robot parameters (for simulation use `scripts/cache/`)

## Installation

This code was written for Python 2.7, because that is compatible with the Sawyer robot

### Dependencies
Dependencies are tracked in `requirements.txt` and `requirements_windows.txt`.  Install with:
```
pip install --r [requirements.txt|requirements_windows.txt]
pip3 install --r requirements_python3.txt
```

### For use with the robot:
```
sudo apt install gphoto2 libgphoto2*

sudo apt-get install dos2unix
dos2unix cache/*
```
