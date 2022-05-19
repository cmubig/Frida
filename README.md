# SawyerPainter


```
cd SawyerPainter/scripts/
python paint.py --use_cache --simulate --cache_dir cache --target frida.jpg

# In another terminal
tensorboard --logdir SawyerPainter/scripts/painting
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
```

### For use with the robot:
```
sudo apt install gphoto2 libgphoto2*

sudo apt-get install dos2unix
dos2unix cache/*
```