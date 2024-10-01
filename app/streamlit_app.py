import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os

# Load fine-tuned model
@st.cache_resource
def load_model():

    model_path = "outputs/fine_tuned_models/"
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    return pipeline

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
