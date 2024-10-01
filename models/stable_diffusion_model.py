import torch
from diffusers import StableDiffusionPipeline

class StableDiffusionModel:
    def __init__(self, device='cuda'):
        self.device = device
        # Load the Stable Diffusion model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(self.device)
    
    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        with torch.autocast(self.device):
            image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        return image
