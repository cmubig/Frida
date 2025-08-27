# python3 paint.py --objective clip_conv_loss --objective_data ~/Downloads/carnegie_11_85.jpg --objective_weight 1.0 --num_strokes 81 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush    --robot xarm --use_colors_from /home/robot_painting/Documents/Frida/src/4grey.png --n_colors 4 --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_paint.json  --dont_retrain_stroke_model

# python3 paint.py --simulate --objective clip_conv_loss --objective_data images/arya111.jpg --objective_weight 1.0 --num_strokes 50 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush --robot xarm --ink  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_paint.json --dont_retrain_stroke_model

# python3 paint.py --objective clip_conv_loss --objective_data images/labs_portraits/hakunz.png --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --painting_path images/labs_portraits/hakunz.pkl


# python3 paint.py --objective clip_conv_loss --objective_data images/labs_portraits/Jua.png --objective_weight 1.0 --num_strokes 10 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/circular.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model


# --simulate
# --dont_retrain_stroke_model
# --painting_path


#!/bin/bash

# png_files=(
#     abucker.png
#     acanovil.png
#     bstoler.png
#     chan.png
#     chengyaz.png
#     eliotx.png
#     feiyuz.png
#     haokunz.png
#     hpark3.png
#     hyaejino.png
#     ingridn.png
#     jaeyoons.png
#     jmf1.png
#     jon_arriza.png
#     Jua.png
#     portegak.png
#     pschalde.png
#     sunyuw.png
#     ushin.png
#     uyoo.png
#     vihaan.png
#     zhixuan2.png
#     minkih.png
#     sieunc.png
#     junseoki.png
#     juanalvarez.png
#     seungbel.png
# )

# color_dir=(
#     colors_1
#     colors_2
#     colors_3
# )
#  # Loop through image files and generate matching .pkl names
#  subdir="labs_portraits"
#  for sub_dir in "${color_dir[@]}"; do
#    for img in "${png_files[@]}"; do
#      img_name="${img%.*}"
#      pkl_file="$img_name.pkl"
#      echo "Processing $img with corresponding $pkl_file"
#      python3 paint.py --simulate --objective clip_conv_loss --objective_data "images/$subdir/$img" --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/$sub_dir.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model --painting_path "images/$sub_dir/$pkl_file"
#    done
#  done

img_name="seungbel"
subdir="labs_portraits"
sub_dir="colors_3"
python3 paint.py --objective clip_conv_loss --objective_data "images/$subdir/$img_name.png" --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/demo --robot xarm --use_colors_from brush_colors/$sub_dir.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json  --painting_path "images/$sub_dir/$img_name.pkl"
