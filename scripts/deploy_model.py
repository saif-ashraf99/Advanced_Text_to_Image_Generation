import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
import asyncio

# Define the request model
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

# Initialize the FastAPI app
app = FastAPI(title="Stable Diffusion Text-to-Image API")

# Load the Stable Diffusion model asynchronously
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Stable Diffusion model...")

# Load the model in a background task
@app.on_event("startup")
async def load_model():
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    print("Model loaded.")

# Define the API endpoint
@app.post("/generate-image")
async def generate_image(request: GenerateRequest):
    try:
        prompt = request.prompt
        num_inference_steps = request.num_inference_steps
        guidance_scale = request.guidance_scale

        # Generate image
        with torch.autocast(device):
            image = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            )

        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image_base64": img_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    uvicorn.run("deploy_model:app", host="0.0.0.0", port=8000)
