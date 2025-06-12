import os
import json
import random
import argparse
from PIL import Image
import re
from tqdm import tqdm

# Optional imports for model
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def generate_random_bbox(image_width, image_height):
    x1 = random.randint(0, image_width - 1)
    y1 = random.randint(0, image_height - 1)
    x2 = random.randint(x1, image_width)
    y2 = random.randint(y1, image_height)
    return [x1, y1, x2, y2]

def inference_qwen(model, processor, text, image_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"In this UI screenshot, what is the position of the element corresponding to the command \"{text}\" (with bbox)?"}
        ],
    }]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)

def parse_output(text):
    try:
        box = re.findall('(?<=<\|box_start\|>)(.*?)(?=<\|box_end\|>)', text[0])[0]
        numbers = re.findall(r'\d+', box)
        return [int(n) for n in numbers]
    except:
        return None

def process_annotations(args):
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model if needed
    model, processor = None, None
    if args.mode in ["qwen", "os-atlas"]:
        print("Loading model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_path)

    for filename in os.listdir(args.annotation_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(args.annotation_dir, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []

        for entry in tqdm(data, desc=f"Processing {filename}"):
            img_filename = entry["img_filename"]
            img_path = os.path.join(args.image_dir, img_filename)

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            image_width, image_height = entry["img_size"]
            img = Image.open(img_path)
            base_filename = os.path.splitext(os.path.basename(img_filename))[0]
            save_dir = os.path.join(args.output_dir, os.path.dirname(img_filename))
            os.makedirs(save_dir, exist_ok=True)

            # Default init
            bbox = None
            parsed_bbox = None
            crop_filename = None
            cropped_path = None
            cropped_coord = None

            if args.mode == "random":
                bbox = generate_random_bbox(image_width, image_height)

            elif args.mode in ["qwen", "os-atlas"]:
                instruction = entry["instruction"]
                model_result = inference_qwen(model, processor, instruction, img_path)
                parsed_bbox = parse_output(model_result)

                def is_valid(parsed_bbox):
                    if not parsed_bbox or len(parsed_bbox) != 4:
                        return False
                    cx = (parsed_bbox[0] + parsed_bbox[2]) // 2
                    cy = (parsed_bbox[1] + parsed_bbox[3]) // 2
                    return 0 <= cx <= 1000 and 0 <= cy <= 1000  # normalized 0~1000

                if is_valid(parsed_bbox):
                    if img_filename.endswith("_null.png"):
                        bbox = [
                            int(parsed_bbox[0] * image_width / 1000),
                            int(parsed_bbox[1] * image_height / 1000),
                            int(parsed_bbox[2] * image_width / 1000),
                            int(parsed_bbox[3] * image_height / 1000)
                        ]
                    else:
                        bbox = [
                            int(parsed_bbox[0] * args.crop_size / 1000),
                            int(parsed_bbox[1] * args.crop_size / 1000),
                            int(parsed_bbox[2] * args.crop_size / 1000),
                            int(parsed_bbox[3] * args.crop_size / 1000)
                        ]
                else:
                    # Parsing 실패 or bbox 비정상 → _null 처리
                    crop_filename = f"{base_filename}_null.png"
                    cropped_path = os.path.join(save_dir, crop_filename)
                    img.save(cropped_path)

                    entry["cropped_img"] = cropped_path
                    entry["cropped_img_coord"] = None
                    entry["real_coord_output"] = None
                    entry["model_output"] = None
                    results.append(entry)
                    continue

                entry["model_output"] = parsed_bbox

            entry["real_coord_output"] = bbox

            # bbox 존재 여부에 따라 crop 처리
            def is_bbox_invalid(bbox):
                if not bbox or len(bbox) != 4:
                    return True
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                return cx < 0 or cy < 0 or cx > image_width or cy > image_height

            if is_bbox_invalid(bbox):
                crop_filename = f"{base_filename}_null.png"
                cropped_path = os.path.join(save_dir, crop_filename)
                img.save(cropped_path)

                entry["cropped_img"] = cropped_path
                entry["cropped_img_coord"] = None
                entry["real_coord_output"] = None
                entry["model_output"] = None
            else:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                left = max(0, min(cx - args.crop_size // 2, image_width - args.crop_size))
                top = max(0, min(cy - args.crop_size // 2, image_height - args.crop_size))
                right = left + args.crop_size
                bottom = top + args.crop_size
                cropped_img = img.crop((left, top, right, bottom))

                crop_filename = f"{base_filename}_{left}_{top}_{right}_{bottom}.png"
                cropped_path = os.path.join(save_dir, crop_filename)
                cropped_img.save(cropped_path)

                cropped_coord = [left, top, right, bottom]
                entry["cropped_img"] = cropped_path
                entry["cropped_img_coord"] = cropped_coord

            results.append(entry)

        output_json_path = os.path.join(args.output_dir, f"cropped_{filename}")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Saved: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "qwen", "os-atlas"], required=True)
    parser.add_argument("--annotation_dir", required=True, help="Input JSON directory")
    parser.add_argument("--image_dir", required=True, help="Original image directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--model_path", default="OS-Copilot/OS-Atlas-Base-7B", help="Model path for qwen/os-atlas") #"OS-Copilot/OS-Atlas-Base-7B" , "Qwen/Qwen2-VL-7B-Instruct"
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size in pixels")
    args = parser.parse_args()

    process_annotations(args)