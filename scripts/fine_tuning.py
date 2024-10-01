import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import mlflow
import mlflow.pytorch

# Configuration
model_name = "CompVis/stable-diffusion-v1-4"
output_dir = "outputs/fine_tuned_models/"
epochs = 3
batch_size = 1  # Stable Diffusion requires significant VRAM
learning_rate = 5e-6
image_size = 512  # Stable Diffusion uses 512x512 images

# Initialize MLflow
mlflow.set_experiment("Stable Diffusion Fine-Tuning")

# Initialize Accelerator
accelerator = Accelerator()

# Load Pretrained Stable Diffusion Model
pipeline = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=True  # Ensure you have access rights
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# Move pipeline to device
pipeline.to(accelerator.device)

# Load Dataset
def preprocess_captions(caption):
    # Implement any necessary caption preprocessing
    return caption

dataset = load_dataset(
    'imagefolder',
    data_dir='data/processed/images/',
    split='train',
    cache_dir='data/cache/'
)

def transforms(examples):
    images = [image.convert("RGB").resize((image_size, image_size)) for image in examples['image']]
    captions = [preprocess_captions(caption) for caption in examples['text']]
    return {'images': images, 'captions': captions}

dataset = dataset.with_transform(transforms)

dataloader = DataLoader(dataset, batch_size=batch_size)

# Training Loop
with mlflow.start_run():
    mlflow.log_params({
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(pipeline.unet):
                images = batch['images']
                captions = batch['captions']

                # Prepare inputs
                latents = pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor

                # Get text embeddings
                text_input = pipeline.tokenizer(
                    captions, padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
                )
                text_embeddings = pipeline.text_encoder(
                    text_input.input_ids.to(accelerator.device)
                )[0]

                # Get noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()

                # Get noisy latents
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual
                noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

                # Compute loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # Backpropagation
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                if step % 10 == 0:
                    mlflow.log_metric("loss", loss.item(), step=epoch * len(dataloader) + step)
                    print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

    # Save the fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_pretrained(output_dir)
    mlflow.pytorch.log_model(pipeline, "stable_diffusion_model")
