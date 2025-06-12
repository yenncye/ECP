from typing import  Union, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2VL:
    def __init__(self, model_path, device_map='auto', low_mem=False):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype='auto', device_map=device_map
        )
        min_pixels = 256*28*28
        max_pixels = 960*28*28
        if low_mem: self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        else: self.processor = AutoProcessor.from_pretrained(model_path)
    
    def inference(
        self,
        text: str,
        image: Union[str, list[str]],
        image_caption: None,
        image_option: None,
    ):
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        if type(image) == str:
            messages[0]['content'].append({"type": "image", "image": image})
        else:
            for i, img in enumerate(image):
                if image_caption is None: messages[0]['content'].append({"type": f"Image-{i}\n"})
                else: messages[0]['content'].append({"type": f"{image_caption[i]}\n"})
                messages[0]['content'].append(({"type": "image", "image": img}))

        messages[0]['content'].append({"type": "text", "text": text})
        
        if image_option is not None:
            img_cnt = 0
            for i in range(len(messages[0]['content'])):
                if messages[0]['content'][i]['type'] == 'image':
                    messages[0]['content'][i] = dict(messages[0]['content'][i], **image_option[img_cnt])
                    img_cnt += 1
                    

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=128, 
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] 
            for in_ids, out_ids in zip(inputs.input_ids.repeat((len(generated_ids)//len(inputs.input_ids), 1)), generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

    def __call__(
        self,
        text: str,
        image: str,
        image_caption: Union[str, list[str]] = None,
        image_option: Optional[dict] = None,
    ):
        return self.inference(text, image, image_caption, image_option)
