# Template 
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 500 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg

// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 1000 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 200 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T002/

// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 50 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T003/

# Increase image guidance for medium --> 2
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 50 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T004/

# Increase image guidance for medium --> 1.7
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 10 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T005/

# Increase image guidance for medium --> 1.7 --> Rerun with 100 iters because we need more for iterative planning and image generation
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 100 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T006/

# Running T006 params with 1000 opt iterations and render height 200
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 1000 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 200 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T007/

# Trying to rerun T007 with increased border and 100 strokes
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 1000 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 200 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T008/