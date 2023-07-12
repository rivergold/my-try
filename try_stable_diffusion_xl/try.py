from diffusers import DiffusionPipeline
import torch

if __name__ == '__main__':

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-0.9",
        cache_dir='',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    prompt = "An astronaut riding a green horse"

    image = pipe(prompt=prompt).images[0]
    image.save('./res.png')