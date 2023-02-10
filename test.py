from diffusers import StableDiffusionPipeline
import torch

model_id = "andite/anything-v4.0"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")

prompt = "cat"
image = pipe(prompt).images[0]

image.save("./hatsune_miku.png")
