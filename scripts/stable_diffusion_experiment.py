import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the Stable Diffusion model
model = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

# Function to generate and display image
def generate_and_display(prompt, num_inference_steps=50, guidance_scale=7.5):
    with torch.autocast(device):
        image = model(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    plt.imshow(image)
    plt.axis('off')
    plt.title(prompt)
    plt.show()

# Test the model with a sample prompt
if __name__ == "__main__":
    prompt = "A cat sitting on a park bench"
    generate_and_display(prompt)
