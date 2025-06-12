import os.path as osp
import io, base64
import pandas as pd
import argparse
import json
import re
from PIL import Image
from tqdm import tqdm
from utils.model import Qwen2VL

def make_prompt(data, image):
    w, h = image.size 
    question = data['question']
    a = data['A']
    b = data['B']
    c = data['C']
    d = data['D']
    prompt = (
        f"Given an image of size ({w}, {h}) and the following multiple-choice question:\n"
        f"Question: {question} Choose answer among choice A, B, C and D\n"
        f"A: {a}\n"
        f"B: {b}\n"
        f"C: {c}\n"
        f"D: {d}\n"
        "Identify the most important object in the image necessary to answer the question.\n"
        "Output Format: (x, y)\n"
        "Both x and y must be strictly normalized from 0 to 1000, regardless of the original image dimensions.\n"
        "The output must be a single coordinate point (x, y) representing the most relevant object.\n"
    )
        
    return prompt

def inference(model:Qwen2VL, data):
    image = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    image.save(f'tmp_stage1.png')
    prompt = make_prompt(data, image)
    output = model(prompt, f'tmp_stage1.png')
    return output

def parse_point(text):
    try:
        answer = re.findall(r'\((.*?)\)', text)
        x, y = [num.replace('x', '').replace('y', '') for num in answer[-1].split(',')]
        if 0 <= float(x) <= 1 and 0 <= float(y) <= 1:
            return [float(x), float(y)]
        return [float(x)/1000, float(y)/1000]
    except:
        return None

def main(args):
    dataset = pd.read_parquet(args.data, engine='pyarrow')
    model = Qwen2VL(args.model_path, low_mem=True)

    with open(osp.join('result', args.output), 'w') as f:
        for i in tqdm(range(len(dataset))):
            ans = {}
            data = dataset.loc[i]
            ans['index'] = int(data['index'])
            ans['category'] = str(data['category'])
            ans['cycle_category'] = str(data['cycle_category'])
            ans['answer'] = str(data['answer'])

            output = inference(model, data)[0]
            model_answer = parse_point(output)
            ans['stage1_output'] = model_answer

            f.write(json.dumps(ans)+'\n') 
            f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='HR-Bench/hr_bench_8k.parquet')
    parser.add_argument('--model_path', default='/root/models/Qwen2-VL-7B-Instruct')
    parser.add_argument('--output', default='hrbench_stage1.jsonl')
    args = parser.parse_args()
    main(args)