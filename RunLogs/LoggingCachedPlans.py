# Template 
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 500 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg

// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 1000 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 200 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T002/

// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 50 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T003/

# Increase image guidance for medium
// python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 50 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --render_height 128 --background_image ../cofrida/blank_canvas.jpg --save_dir=/scratch/tshankar/CoachFrida/SavedPaintings/T004/

