import os
import json
import re
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def inference(model, processor, text, image_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"In this UI screenshot, what is the position of the element corresponding to the command \"{text}\" (with bbox)?"},
        ]
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)

def parse(text):
    try:
        box = re.findall(r'(?<=<\|box_start\|>)(.*?)(?=<\|box_end\|>)', text[0])[0]
        numbers = re.findall(r'\d+', box.replace('(', ' ').replace(')', ' ').replace(',', ' '))
        return [int(n) for n in numbers]
    except:
        return None

def is_valid_bbox(bbox, width, height):
    if not bbox or len(bbox) != 4:
        return False
    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    return 0 <= cx <= width and 0 <= cy <= height

def run_second_stage(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    output_base = os.path.join("./P_output", os.path.basename(args.first_stage_dir), args.second_stage_model)
    os.makedirs(output_base, exist_ok=True)

    for file in os.listdir(args.first_stage_dir):
        if not file.endswith(".json"): continue
        input_path = os.path.join(args.first_stage_dir, file)
        output_path = os.path.join(output_base, f"inference_{file}")

        with open(input_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in tqdm(entries, desc=f"Processing {file}"):
            img_path = entry["cropped_img"]
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            instruction = entry["instruction"]
            result = inference(model, processor, instruction, img_path)
            entry["in_cropimg_model_output"] = result

            parsed_bbox = parse(result)
            if parsed_bbox:
                if img_path.endswith("_null.png"):
                    width, height = entry["img_size"]
                    scaled = [int(parsed_bbox[i] * width / 1000 if i % 2 == 0 else parsed_bbox[i] * height / 1000) for i in range(4)]
                else:
                    scaled = [int(n * args.crop_size / 1000) for n in parsed_bbox]
            else:
                scaled = None

            if scaled and is_valid_bbox(scaled, *entry["img_size"]):
                entry["in_cropimg_coord_output"] = scaled

                # 원본 좌표로 복원
                if entry.get("cropped_img_coord"):
                    left, top, _, _ = entry["cropped_img_coord"]
                    entry["final_output"] = [
                        scaled[0] + left,
                        scaled[1] + top,
                        scaled[2] + left,
                        scaled[3] + top,
                    ]
                else:
                    entry["final_output"] = None
            else:
                entry["in_cropimg_model_output"] = None
                entry["in_cropimg_coord_output"] = None
                entry["final_output"] = None

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4, ensure_ascii=False)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_stage_dir", required=True, help="Directory of first-stage output (random/qwen/os-atlas)")
    parser.add_argument("--second_stage_model", required=True, choices=["qwen", "os-atlas"], help="Second-stage model name")
    parser.add_argument("--model_path", required=True, help="Path to the Qwen or OS-Atlas model")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size used in first-stage cropping")
    args = parser.parse_args()
    run_second_stage(args)
