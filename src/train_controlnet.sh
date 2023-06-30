// export MODEL_DIR="/home/frida/Downloads/frida_sd_fine_tune_attempt1-20230302T170503Z-001/frida_sd_fine_tune_attempt1/"

export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./controlnet_models/"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --data_dict '/home/frida/paint/FridaXArm/src/lora_data_200/data_dict.pkl' \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
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
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --tracker_project_name="add_strokes6_blanks" \
 --validation_steps=400 \
 --num_train_epochs=25 \
 --use_8bit_adam \
 --proportion_empty_prompts 0.1