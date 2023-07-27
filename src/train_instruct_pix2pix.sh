// export MODEL_DIR="/home/frida/Downloads/frida_sd_fine_tune_attempt1-20230302T170503Z-001/frida_sd_fine_tune_attempt1/"
# conda activate frida
export MODEL_DIR="timbrooks/instruct-pix2pix"
export OUTPUT_DIR="./controlnet_models/"

accelerate launch train_instruct_pix2pix.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --data_dict '/home/frida/paint/FridaControlNet2/src/controlnet_data_ink_3method2/data_dict.pkl' \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --tracker_project_name="pix2pix_ink_3methods2" \
 --validation_steps=400 \
 --num_train_epochs=100 \
 --validation_image "/home/frida/paint/FridaXArm/src/lora_quality_data/0/id36_pass0_100strokes.jpg" \
                    "/home/frida/paint/FridaXArm/src/lora_quality_data/0/id36_pass0_100strokes.jpg" \
                    "/home/frida/paint/FridaXArm/src/lora_data_200/0/id15_start.jpg" \
                    "/home/frida/paint/FridaXArm/src/lora_data_200/0/id11_start.jpg" \
                    "/home/frida/Downloads/20230525_121903.jpg" \
                    "/home/frida/Downloads/20230525_121903.jpg" \
                    "/home/frida/paint/FridaXArm/src/lora_quality_data/1/id100_start.jpg" \
                    "/home/frida/paint/FridaXArm/src/lora_quality_data/1/id100_start.jpg" \
                    "/home/frida/Downloads/rex.jpg" \
                    "/home/frida/Downloads/rex.jpg" \
 --validation_prompt "A sketch of a girl with the eiffel tower in the background" \
                    "" \
                    "Albert Einstein with his arms crossed" \
                    "A piece of pie in space" \
                    "a drawing of a person with a triangle for a body and a smiling face" \
                    "A drawing of a turtle" \
                    "A frog astronaut" \
                    "Albert Einstein Dancing" \
                    "A drawing of a dinosaur with a clown body" \
                    "A drawing of a dinosaur" \
 --use_8bit_adam \
 --num_validation_images=2 \
 --seed 0