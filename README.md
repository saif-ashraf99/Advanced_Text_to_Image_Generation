# Advanced Text-to-Image Generation Project with Stable Diffusion

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up the Environment](#2-set-up-the-environment)
  - [3. Run the Streamlit App](#3-run-the-streamlit-app)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Streamlit Web Application](#streamlit-web-application)
- [Project Details](#project-details)
  - [Model Implementation](#model-implementation)
  - [Streamlit Integration](#streamlit-integration)
  - [Licensing Considerations](#licensing-considerations)
- [Project Structure](#project-structure-detailed)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The **Advanced Text-to-Image Generation Project** leverages the power of **Stable Diffusion**, a state-of-the-art latent diffusion model, to generate high-quality images from textual descriptions. The project includes a user-friendly **Streamlit** web application for interactive image generation, making advanced AI accessible to users without deep technical expertise.

## Features

- **High-Quality Image Generation**: Generate detailed and realistic images from text prompts using Stable Diffusion.
- **Interactive Web Application**: Use the Streamlit app to generate images in real-time with an intuitive interface.
- **Command Line Interface**: Generate images directly from the command line for integration into other workflows.
- **Experimentation Notebook**: Explore and experiment with Stable Diffusion in a Jupyter notebook.
- **MLOps Integration**: Optional MLflow integration for experiment tracking and model management.

## Requirements

- **Python 3.8 or higher**
- **CUDA-compatible GPU** (recommended for optimal performance)
- **Python Packages**: Listed in `requirements.txt`

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/saif-ashraf99/Advanced_Text_to_Image_Generation.git
cd Advanced_Text_to_Image_Generation
```

### 2. Set Up the Environment

It's recommended to use a virtual environment to manage dependencies.

#### Using `venv`

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### Install Dependencies

Upgrade `pip` and install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Some packages may require additional system-level dependencies. If you encounter issues, refer to the package documentation.

### 3. Run the Streamlit App

After setting up the environment, you can run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

- Access the app in your web browser at `http://localhost:8501`.
- Input a text prompt, and the app will display the generated image.

## Usage

### Command Line Interface

You can generate images directly using the command line script:

```bash
python scripts/generate_images.py --text "A futuristic cityscape at night"
```

- **Arguments**:
  - `--text`: The text prompt for image generation.
  - `--output_dir` (optional): Directory to save the generated images. Default is `outputs/generated_images/`.

### Streamlit Web Application

The Streamlit app provides an interactive interface:

1. Run the app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Enter your text prompt in the input box.

4. Click on the "Generate Image" button to see the generated image.

## Project Details

### Model Implementation

- **Stable Diffusion**: Utilizes the `diffusers` library by Hugging Face to access the Stable Diffusion model (`v1.5`).
- **Model Wrapper**: Implemented in `models/stable_diffusion_model.py` for easy integration.
- **Device Support**: Supports both GPU and CPU, but GPU is recommended for performance.

### Streamlit Integration

- **App Location**: `app/streamlit_app.py`
- **Features**:
  - Text input for prompts.
  - Real-time image generation and display.
  - Option to save generated images.
- **Caching**: The model is cached using `@st.cache_resource` to prevent reloading on every interaction.

### Licensing Considerations

- **Stable Diffusion License**: Released under the [CreativeML Open RAIL++-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
- **Usage Restrictions**:
  - **Allowed**: Personal and commercial use, with proper attribution.
  - **Disallowed**: Generation of illegal, harmful, or NSFW content.
- **Compliance**: Users are responsible for ensuring that they comply with the license terms when using this project.

## Project Structure (Detailed)

```
Advanced_Text_to_Image_Generation/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── models/
│   └── stable_diffusion_model.py # Stable Diffusion model wrapper
├── notebooks/
│   └── stable_diffusion_experiments.ipynb # Jupyter notebook for experimentation
├── outputs/
│   └── generated_images/         # Directory for saving generated images
├── scripts/
│   ├── generate_images.py        # CLI script for image generation
│   ├── download_data.py          # (Optional) Script to download datasets
│   └── preprocess_data.py        # (Optional) Script for data preprocessing
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # Project license
```

- **`app/`**: Contains the Streamlit application code.
- **`models/`**: Includes model implementations.
- **`notebooks/`**: Contains Jupyter notebooks for experiments.
- **`outputs/`**: Stores generated images and other outputs.
- **`scripts/`**: Utility scripts for various tasks.
- **`requirements.txt`**: Lists all required Python packages.
- **`README.md`**: This file.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right of the repository page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/saif-ashraf99/Advanced_Text_to_Image_Generation.git
   ```

3. **Create a New Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**: Implement your feature or fix.

5. **Commit Your Changes**:

   ```bash
   git commit -am 'Add a descriptive commit message'
   ```

6. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**: Go to the original repository and create a pull request from your fork.

**Note**: The Stable Diffusion model used in this project is released under the [CreativeML Open RAIL++-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Users must comply with its terms in addition to the project's MIT License.

## Acknowledgments

- **Stability AI and RunwayML**: For developing and releasing Stable Diffusion.
- **Hugging Face**: For the `diffusers` library and support.
- **Streamlit**: For providing an easy way to build interactive web apps.

---

**Disclaimer**: This project is intended for educational and research purposes. I am not responsible for any misuse of the generated content. Users must adhere to the licensing terms and ensure ethical use.

# Final Notes

Thank you for your interest in the **Advanced Text-to-Image Generation Project with Stable Diffusion**. I hope this project provides valuable insights and tools for your text-to-image generation needs. If you have any questions or need assistance, feel free to open an issue or contact the project maintainers.

Happy generating!
