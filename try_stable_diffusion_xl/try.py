from datetime import datetime
from pathlib import Path
import traceback
from diffusers import DiffusionPipeline
import torch

if __name__ == '__main__':

    pipe = DiffusionPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-xl-base-0.9",
        '/home/hejing/data/opensource/huggingface/model/model--stabilityai--stable-diffusion-xl-base-0.9',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    prompt = "An astronaut riding a green horse"

    while True:
        prompt = input('prompt:')
        neg_prompt = input('neg_prompt:')
        print(len(neg_prompt))
        imgs = []
        try:
            res = pipe(prompt=prompt,
                       negative_prompt=neg_prompt,
                       num_images_per_prompt=2)
            print(type(res))
            imgs = res.images
        except Exception as e:
            print(traceback.format_exc())
        cur_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_dir = Path('./out') / cur_time_str
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(imgs):
            img.save(f"{out_dir}/{idx}.png")
