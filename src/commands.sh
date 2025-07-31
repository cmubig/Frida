# python3 paint.py --objective clip_conv_loss --objective_data ~/Downloads/carnegie_11_85.jpg --objective_weight 1.0 --num_strokes 81 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush    --robot xarm --use_colors_from /home/robot_painting/Documents/Frida/src/4grey.png --n_colors 4 --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_paint.json  --dont_retrain_stroke_model

# python3 paint.py --simulate --objective clip_conv_loss --objective_data images/arya111.jpg --objective_weight 1.0 --num_strokes 50 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush --robot xarm --ink  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_paint.json --dont_retrain_stroke_model

# python3 paint.py --objective clip_conv_loss --objective_data images/labs_portraits/hakunz.png --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --painting_path images/labs_portraits/hakunz.pkl


# python3 paint.py --objective clip_conv_loss --objective_data images/labs_portraits/Jua.png --objective_weight 1.0 --num_strokes 10 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/circular.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model


# --simulate
# --dont_retrain_stroke_model
# --painting_path


#!/bin/bash

#  files=(
#    abucker.png
#    bstoler.png
#    chan.png
#    chengyaz.png
#    eliotx.png
#    feiyuz.png
#    haokunz.png
#    hpark3.png
#    hyaejino.png
#    ingridn.png
#    jmf1.png
#    Jua.png
#    portegak.png
#    pschalde.png
#    sunyuw.png
#    ushin.png
#    uyoo.png
#    vihaanm.png
#    zhixuan2.png
#    jaeyoons.png
#    jon_arriza.png
#  )

#  # Loop through image files and generate matching .pkl names
#  for img in "${files[@]}"; do
#    img_name="${img%.*}"
#    pkl_file="$img_name.pkl"
#    echo "Processing $img with corresponding $pkl_file"
#    python3 paint.py --simulate --objective clip_conv_loss --objective_data "images/labs_portraits/$img" --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model --painting_path "images/labs_portraits/$pkl_file"
#  done

img_name="jon_arrizahaokunz"
python3 paint.py --objective clip_conv_loss --objective_data "images/labs_portraits/$img_name.png" --objective_weight 1.0 --num_strokes 80 --lr_multiplier 2.5 --init_optim_iter 2000 --num_adaptations 1 --use_cache --cache_dir caches/small_brush_holder --robot xarm --use_colors_from brush_colors/random.png --n_colors 4  --xarm_ip 192.168.1.168 --vae_path mocap/saved_models/general.pt --materials_json ../materials_xarm_holder.json --dont_retrain_stroke_model --painting_path "images/labs_portraits/$img_name.pkl"