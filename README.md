# Text2ImageWebApp

A web application for generating high-quality images from text descriptions using Stable Diffusion models. Built with Gradio for an intuitive web interface and optimised to run efficiently on consumer hardware without requiring expensive GPU resources.

🌟 Features

Core Functionality

Text-to-Image Generation: Transform detailed text prompts into stunning visual artwork

Multiple Model Support: Choose between Stable Diffusion 1.5 and 2.1 models

CPU Optimisation: Specially tuned to run efficiently on CPU-only systems

Style Presets: Built-in artistic styles (Photorealistic, Digital Art, Anime, Oil Painting, Watercolour)

Batch Generation: Create multiple variations with a single click

Advanced Controls

Seed Control: Reproducible results with manual or automatic seed selection

Generation Parameters: Fine-tune steps, guidance scale, and image dimensions

Negative Prompts: Specify what to avoid in generated images

Smart Parameter Adjustment: Automatic optimisation for CPU performance

User Experience

Intuitive Web Interface: Clean, modern Gradio-based UI accessible via web browser

Generation History: Keep track of recent creations with metadata

System Monitoring: Real-time memory usage tracking

Comprehensive Help: Built-in tips and best practices

Progress Tracking: Real-time status updates during generation                                                                    
🚀 Quick Start

Prerequisites

Python 3.8 or higher

8GB RAM minimum (16GB recommended)

15GB free disk space (for models and generated images)

Stable internet connection for initial model download

Installation

Clone or download this repository

git clone <Text2ImageWebApp>

cd Text2Image-Web-App

Create a virtual environment (recommended)

python -m venv sd_env

source sd_env/bin/activate  # On Windows: sd_env\Scripts\activate

Install dependencies

pip install -r requirements.txt

Run the application

python app.py

Open your browser and navigate to the URL in the terminal.                                                                           
Happy generating! 🎨

Transform your imagination into stunning visual art with the power of AI, optimised to run on your hardware.
