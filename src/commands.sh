python3 codraw.py --use_cache --cache_dir caches/mars_sharpie_film/ --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 1.3 --optim_iter 500 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model


python3 codraw.py --use_cache --cache_dir caches/vae_sharpie_final --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 0.3 --optim_iter 200 --ink  --xarm_ip 192.168.2.157  --dont_retrain_stroke_model --n_predicted_strokes 32 --continue_training /home/frida/Documents/RLFrida/FridaTransformer/src/stroke_predictor_models/08_15__11_25_08/stroke_predictor_weights.pth --num_prediction_rounds 2 --vae_path mocap/saved_models/general.pt

python3 codraw.py --use_cache --cache_dir caches/vae_sharpie_final --materials_json ../materials_mars_8x8.json --cofrida_model skeeterman/CoFRIDA-Sharpie --robot xarm --lr_multiplier 0.4 --optim_iter 200 --ink  --xarm_ip 192.168.1.168  --dont_retrain_stroke_model --vae_path mocap/saved_models/general.pt

tensorboard --logdir painting_log