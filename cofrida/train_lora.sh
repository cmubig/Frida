
// export MODEL_DIR="CompVis/stable-diffusion-v1-4"
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./lora_models/"

accelerate launch --mixed_precision="fp16"  train_lora.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --data_dict 'controlnet_data_ink_quality_100_vetted3/data_dict.pkl' \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --validation_prompt "A frog astronaut." "The pittsburgh skyline" "A drawing of the Pittsburgh skyline" \
        "A robot playing the piano" "An avocado chair" "Albert Einstein dancing" \
  --validation_steps=100 \
  --tracker_project_name="lora_sketch_vetted3_4" \
  --num_validation_images 6 \
  --num_train_epochs=100 \
  --seed=1337