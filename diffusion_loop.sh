#!/bin/bash

# Directory to loop through
IMG_DIR="/home/frida/bwoodwar/Frida/data/celeba/img_align_celeba/img_align_celeba"
TARGET_DIR="/home/frida/bwoodwar/Frida/data/celeba_diffusion"
COLORS_DIR="/home/frida/bwoodwar/Frida/src/4grey.png"

# Loop through all JPGs
for img in "$IMG_DIR"/*.jpg; do

    img_name=$(basename "$img" .jpg)
    TARGET_SUBDIR="$TARGET_DIR/$img_name"


    # Check if subfolder exists, if so: skip
    if [ -d "$TARGET_SUBDIR" ]; then
        echo "Skipping $img_name (already processed)"
        continue
    fi
    
    mkdir -p "$TARGET_SUBDIR"
    echo "Processing: $img"

    python3 /home/frida/bwoodwar/Frida/src/paint.py --simulate \
            --render_height 256 \
            --use_cache \
            --cache_dir caches/small_brush  \
            --dont_retrain_stroke_model \
            --objective clip_conv_loss \
            --objective_data "$img"  \
            --objective_weight 1.0 \
            --lr_multiplier 1.5 \
            --num_strokes 81 \
            --init_optim_iter 1000 \
            --use_colors_from "$COLORS_DIR" \
            --save_diffusion_data \
            --diffusion_data_dir "$TARGET_SUBDIR"

done