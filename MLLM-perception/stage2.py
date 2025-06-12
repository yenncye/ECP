import os
from PIL import Image
import os.path as osp
import io
import base64
import argparse
import pandas as pd
import json
from utils.model import Qwen2VL
from tqdm import tqdm
import random
Image.MAX_IMAGE_PIXELS = None

def get_point_window(
    w: int,
    h: int,
    center: list[int],
    window_w: int = 672,
    window_h: int = 672,
) -> list[int]:
    x1 = max(0, center[0]-window_w//2-max(0, center[0]+window_w//2-w))
    y1 = max(0, center[1]-window_h//2-max(0, center[1]+window_h//2-h))
    window = [x1, y1, x1+window_w, y1+window_h]
    return window

def slice_image(img_path: str, output_path: str, anchor: list[int]):
    if osp.dirname(output_path) != '' and not osp.exists(osp.dirname(output_path)):
        os.makedirs(osp.dirname(output_path))
    img = Image.open(img_path)
    img = img.crop(anchor)
    img.save(open(output_path, 'wb'))

def make_prompt(data, image):
    w, h = image.size
    question = data['question']
    a = data['A']
    b = data['B']
    c = data['C']
    d = data['D']
    prompt = (
        "Task: You are given an image and a multiple-choice question related to it. Analyze the image and select the correct answer.\n"
        "Question:\n"
        f"{question}\n"
        "Choices:\n"
        f"A: {a}\n"
        f"B: {b}\n"
        f"C: {c}\n"
        f"D: {d}\n"
        "Image Details:\n"
        f"Image size: ({w}, {h})"
        "Output format:\n"
        "Provide your answer as a single character: one of 'A', 'B', 'C', or 'D'."
    )
    return prompt

def make_box(w, h, point, ratio=0.25):
    if point is None: return None
    box = get_point_window(w, h, point, int(w*ratio), int(h*ratio))
    return box

def inference(model, data, stage1_output):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    except Exception as e:
        print("IMAGE ERROR!")
        return None

    image.save(f'tmp_stage2.png')
    w, h = image.size
    image_option = [
        {'min_pixels': 28*28*256, 'max_pixels': 28*28*960},
        {'min_pixels': 28*28*256, 'max_pixels': 28*28*1280},
    ]
    input_image = [f'tmp_stage2.png', f'tmp_stage2_crop.png']
    input_label = ['Full image: ', 'Cropped image: ']
    try:
        point = [int(stage1_output[0]*w), int(stage1_output[1]*h)]
        box = make_box(w, h, point)
        slice_image(f'tmp_stage2.png', f'tmp_stage2_crop.png', box)
            
        prompt = make_prompt(data, image)
        return model(prompt, input_image, image_caption=input_label, image_option=image_option)[0]
    except Exception as e:
        return None

def main(args):
    dataset = pd.read_parquet(args.data, engine='pyarrow')
    model = Qwen2VL(args.model_path, low_mem = False)
    stage1 = [json.loads(l) for l in open(args.stage1).readlines()] if not args.random else None

    with open(osp.join('result', args.output), 'w') as f:
        for i in tqdm(range(len(dataset))):
            ans = {}
            data = dataset.loc[i]
            ans['index'] = int(data['index'])
            ans['category'] = str(data['category'])
            ans['cycle_category'] = str(data['cycle_category'])
            ans['answer'] = str(data['answer'])

            if args.random:
                stage1_output = [random.randint(0, 1000)/1000, random.randint(0,1000)/1000]
            else:
                stage1_output = stage1[i]['stage1_output']

            output = inference(model, data, stage1_output)
            ans['stage1_output'] = stage1_output
            ans['output'] = output
            f.write(json.dumps(ans)+'\n') 
            f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='HR-Bench/hr_bench_8k.parquet')
    parser.add_argument('--stage1', default='result/hrbench_stage1.jsonl')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--model_path', default='/root/models/Qwen2-VL-7B-Instruct')
    parser.add_argument('--output', default='hrbench_stage2.jsonl')
    args = parser.parse_args()
    main(args)
