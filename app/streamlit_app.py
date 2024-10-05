import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load fine-tuned model
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

pipeline = load_model()

# Streamlit App
st.title("Advanced Text-to-Image Generation with Stable Diffusion")
st.write("Enter a textual description to generate an image.")

# Text input
text_description = st.text_input("Text Description", value="A beautiful landscape with mountains")

if st.button("Generate Image"):
    with st.spinner('Generating image...'):
        # Generate image
        image = pipeline(text_description).images[0]

        # Display image
        st.image(image, caption='Generated Image', use_column_width=True)

        # Save image
        os.makedirs('outputs/generated_images/', exist_ok=True)
        image_path = f"outputs/generated_images/generated_{text_description.replace(' ', '_')}.png"
        image.save(image_path)
        st.success(f"Image saved to {image_path}")
