export MODEL_DIR="timbrooks/instruct-pix2pix"
export OUTPUT_DIR="./cofrida_model_ink/"

accelerate launch train_instruct_pix2pix.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --data_dict './train_data/ink/data_dict.pkl' \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --tracker_project_name="cofrida_log_0" \
 --validation_steps=400 \
 --num_train_epochs=100 \
 --validation_image "./blank_canvas.jpg" \
                    "./blank_canvas.jpg" \
 --validation_prompt "A frog astronaut" \
                     "A drawing of a dinosaur" \
 --use_8bit_adam \
 --num_validation_images=2 \
 --seed 0