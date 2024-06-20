
python3 create_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 500 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate

python3 execute_plan.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json  --robot xarm --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --simulate --saved_plan ./saved_plans/surfer/plan.pt