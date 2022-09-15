python scripts/plan.py \
--simulate --max_height 196 --use_cache --cache_dir cache/ --n_stroke_models 1 \
--init_objective l2 --init_objective_data /mnt/Data1/vmisra/Frida/sample/horse.PNG --init_objective_weight 1.0 \
--objective style text \
--objective_data /mnt/Data1/vmisra/Frida/sample/horse.PNG  "A painting of Van Gogh" \
--objective_weight 0.5 1.0 \
--lr_multiplier 0.5 \
--num_strokes 600 --optim_iter 200 \