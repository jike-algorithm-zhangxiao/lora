#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

conda activate zx

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ubuntu/jc/jeambooth/uploads/63d905fd8db824f963d0663i"
export OUTPUT_DIR="/home/ubuntu/jc/jeambooth/fine-tunes/63d905fd8db824f963d0663i.person/lora"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a lyc person" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000