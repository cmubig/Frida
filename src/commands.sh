python3 paint.py --use_cache --cache_dir caches/mars_brush --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/jj.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 25 --init_optim_iter 400 --lr_multiplier 2 --ink --dont_retrain_stroke_model

python3 paint.py --use_cache --cache_dir caches/mars_brush_lift --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/uksang.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 36 --init_optim_iter 800 --lr_multiplier 5  --dont_retrain_stroke_model --robot xarm --use_colors_from ~/Downloads/4grey.png --n_colors 4

<<<<<<< HEAD
python3 paint.py --use_cache --cache_dir caches/mars_brush_lift --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/uksang.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 36 --init_optim_iter 800 --lr_multiplier 5   --robot xarm --use_colors_from ~/Downloads/4grey.png --n_colors 4 --simulate --dont_retrain_stroke_model

python3 paint.py --use_cache --cache_dir caches/mars_brush_lift2 --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/uksang.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 36 --init_optim_iter 800 --lr_multiplier 5   --robot xarm --use_colors_from ~/Downloads/4grey.png --n_colors 4 --xarm_ip 192.168.1.168



python3 codraw.py --use_cache --cache_dir caches/mars_sharpie2/ --materials_json ../materials_mars_7x7.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.7 --optim_iter 150 --ink --dont_retrain_stroke_model

#webcam interface
python3 paint.py --use_cache --cache_dir caches/mars_brush_lift4 --n_colors 4 --use_colors_from 4grey.png --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data /home/big/imgs/last_portrait.png --objective_weight 1.0  --num_adaptations 1  --num_strokes 36 --init_optim_iter 400 --lr_multiplier 2.5   --robot xarm --xarm_ip 192.168.1.168 --dont_retrain_stroke_model --webcam_interface --save_painting
=======

python3 paint.py --use_cache --cache_dir caches/mars_brush_lift --materials_json ../materials_mars_7x7.json  --objective clip_conv_loss --objective_data ~/Downloads/uksang.jpg --objective_weight 1.0  --num_adaptations 1  --num_strokes 36 --init_optim_iter 800 --lr_multiplier 5   --robot xarm --use_colors_from ~/Downloads/4grey.png --n_colors 4 --simulate --dont_retrain_stroke_model


python3 codraw.py --use_cache --cache_dir caches/mars_sharpie2/ --materials_json ../materials_mars_7x7.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.7 --optim_iter 150 --ink --dont_retrain_stroke_model
>>>>>>> ffd451483ca3dcd3e79e4fd838237c30c732a213
