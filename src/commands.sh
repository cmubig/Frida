# python3 paint.py --objective clip_conv_loss --objective_data ~/Downloads/carnegie_11_85.jpg --objective_weight 1.0 --num_strokes 81 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush    --robot xarm --use_colors_from /home/robot_painting/Documents/Frida/src/4grey.png --n_colors 4 --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_paint.json  --dont_retrain_stroke_model

# python3 paint.py --simulate --objective clip_conv_loss --objective_data images/arya111.jpg --objective_weight 1.0 --num_strokes 50 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush --robot xarm --ink  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_paint.json --dont_retrain_stroke_model

# python3 paint.py --objective clip_conv_loss --objective_data images/labs_portraits/hakunz.png --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --painting_path images/labs_portraits/hakunz.pkl


# python3 paint.py --objective clip_conv_loss --objective_data images/labs_portraits/Jua.png --objective_weight 1.0 --num_strokes 10 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/circular.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model


# --simulate
# --dont_retrain_stroke_model
# --painting_path


#!/bin/bash

#  files=(
#     "IMG_1477 - Janice Min.png"
#     "subject_1.png"
#     "1740717769835-1 - Wons HEE.png"
#     "IMG_3022 - ­이현준 _ 학생 _ 협동과정 인공지능전공.png"
#     "IMG_9679 - 이다연.png"
#     "이은학_증명사진 - 이은학.png"
#  )

#  # Loop through image files and generate matching .pkl names
#  subdir="week_4"
#  for img in "${files[@]}"; do
#    img_name="${img%.*}"
#    pkl_file="$img_name.pkl"
#    echo "Processing $img with corresponding $pkl_file"
#    python3 paint.py --simulate --objective clip_conv_loss --objective_data "images/$subdir/$img" --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model --painting_path "images/$subdir/$pkl_file"
#  done

img_name="acanovil"
subdir="labs_portraits"
python3 paint.py --simulate --objective clip_conv_loss --objective_data "images/$subdir/$img_name.png" --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model --painting_path "images/$subdir/$img_name.pkl"
