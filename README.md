
# 🖼️ Text-to-Image Web App (Stable Diffusion + Gradio)

A web application for generating high-quality images from text descriptions using **Stable Diffusion** models. Built with **Gradio** for an intuitive web interface and optimised to run efficiently on **CPU-only systems**, no expensive GPUs required!

---

## 🌟 Features

### 🎨 Core Functionality
- **Text-to-Image Generation**: Transform detailed text prompts into stunning visual artworks.
- **Multiple Model Support**: Switch between Stable Diffusion **1.5** and **2.1** models.
- **CPU Optimised**: Specially tuned to run on CPU-only setups.
- **Style Presets**: Apply creative presets like *Photorealistic, Digital Art, Anime, Oil Painting, Watercolour*.
- **Batch Generation**: Create multiple image variations in one go.

### ⚙️ Advanced Controls
- **Seed Control**: Ensure reproducibility with manual/auto seed input.
- **Custom Parameters**: Tune inference steps, guidance scale and output resolution.
- **Negative Prompts**: Guide the model to avoid specific elements.
- **Smart Auto-Tuning**: Automatically adjusts size/steps for CPU performance.

### 🧑‍💻 User Experience
- **Modern Gradio UI**: Sleek interface accessible via any browser.
- **Generation History**: View your last 5 creations with prompt/seed info.
- **Memory Monitor**: Track real-time memory usage.
- **Inactivity Handling**: Auto-unloads model after 10 min of idle.
- **Tips & Help Tab**: Best practices and how-to generate better images are built in.

---

## 🚀 Quick Start

### ✅ Prerequisites
- Python 3.8+
- At least 8GB RAM (16GB recommended)
- ~15GB free disk space
- Internet connection (for model download)

---

### 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/saishagoel27/Text2ImageWebApp.git
cd Text2ImageWebApp
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate
# On Windows: .\venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
python app.py
```

5. **Access it via browser**
- Navigate to the **localhost URL** shown in your terminal 

---

## 🎉 Happy Generating!
Unleash your creativity and turn your imagination into breathtaking AI art, directly from your hardware.

---

## 📜 License
This project is licensed under the MIT License.

---


