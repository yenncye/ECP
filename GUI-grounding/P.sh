# Run the second stage inference script 

#!/bin/bash


python Predict.py \
  --first_stage_dir ./EC_output/random \
  --second_stage_model os-atlas \
  --model_path OS-Copilot/OS-Atlas-Base-7B   \
  --crop_size 1024


python Predict.py \
  --first_stage_dir ./EC_output/os-atlas \
  --second_stage_model os-atlas\
  --model_path OS-Copilot/OS-Atlas-Base-7B   \
  --crop_size 1024

python Predict.py \
  --first_stage_dir ./EC_output/qwen \
  --second_stage_model os-atlas \
  --model_path OS-Copilot/OS-Atlas-Base-7B  \
  --crop_size 1024


python Predict.py \
  --first_stage_dir ./EC_output/random \
  --second_stage_model qwen \
  --model_path Qwen/Qwen2-VL-7B-Instruct  \
  --crop_size 1024


python Predict.py \
  --first_stage_dir ./EC_output/os-atlas \
  --second_stage_model qwen \
  --model_path Qwen/Qwen2-VL-7B-Instruct  \
  --crop_size 1024

python Predict.py \
  --first_stage_dir ./EC_output/qwen \
  --second_stage_model qwen \
  --model_path Qwen/Qwen2-VL-7B-Instruct  \
  --crop_size 1024