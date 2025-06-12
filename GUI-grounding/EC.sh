# Run the first stage inference script 

#!/bin/bash


python Extract_Candidate.py \
  --mode random \
  --annotation_dir ./ScreenSpot-Pro/annotations \
  --image_dir ./ScreenSpot-Pro/images \
  --output_dir ./EC_output/random \
  --crop_size 1024

# OS-Atlas Model
python Extract_Candidate.py \
  --mode os-atlas \
  --annotation_dir ./ScreenSpot-Pro/annotations \
  --image_dir ./ScreenSpot-Pro/images \
  --output_dir ./EC_output/os-atlas \
  --model_path OS-Copilot/OS-Atlas-Base-7B \
  --crop_size 1024

# Qwen2VL Model
python Extract_Candidate.py \
  --mode qwen \
  --annotation_dir ./ScreenSpot-Pro/annotations \
  --image_dir ./ScreenSpot-Pro/images \
  --output_dir ./EC_output/qwen \
  --model_path Qwen/Qwen2-VL-7B-Instruct \
  --crop_size 1024

