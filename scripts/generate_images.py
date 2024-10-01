import torch
from diffusers import StableDiffusionPipeline
import argparse
import os

def generate_image(pipeline, text_description):
    with torch.no_grad():
        image = pipeline(text_description).images[0]
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images from text descriptions using Stable Diffusion.')
    parser.add_argument('--text', type=str, required=True, help='Text description for image generation.')
    args = parser.parse_args()

    # Load fine-tuned Stable Diffusion model
    model_path = "outputs/fine_tuned_models/"
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    # Generate image
    image = generate_image(pipeline, args.text)

    # Save image
    os.makedirs('outputs/generated_images/', exist_ok=True)
    image_path = f"outputs/generated_images/generated_{args.text.replace(' ', '_')}.png"
    image.save(image_path)
    print(f"Generated image saved at {image_path}")
