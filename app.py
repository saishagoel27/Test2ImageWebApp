import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import time
import random
import os

# will ensure output directory exists
if not os.path.exists("generated_images"):
    os.mkdir("generated_images")

pipe = None
model_id = "runwayml/stable-diffusion-v1-5"
session_images = []

SD_MODELS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1"
}

STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", photorealistic, 8k, detailed, sharp focus",
    "Digital Art": ", digital art, trending on artstation, highly detailed",
    "Anime": ", anime style, vibrant colors, highly detailed",
    "Oil Painting": ", oil painting, masterpiece, detailed brushwork, textured",
    "Watercolor": ", watercolor painting, artistic, soft colors, flowing"
}

def load_model(model_name):
    global pipe, model_id

    model_id = SD_MODELS.get(model_name, model_id)

    try:
        print(f"Loading model: {model_name}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        )
        scheduler_config = pipe.scheduler.config
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
        pipe = pipe.to("cpu")
        return f"Model '{model_name}' loaded using CPU"
    except Exception as e:
        return f"Error: Failed to load model - {e}"

def generate_image(prompt, negative_prompt, model_name, style_preset,
                   steps, guidance, width, height, seed, use_random_seed):
    global pipe, session_images

    if pipe is None or SD_MODELS.get(model_name) != model_id:
        status = load_model(model_name)
        if status.startswith("Error"):
            return None, [], status

    styled_prompt = prompt + STYLE_PRESETS.get(style_preset, "")

    if use_random_seed:
        used_seed = random.randint(1, 2147483647)
        torch.manual_seed(used_seed)
    else:
        used_seed = int(seed)
        torch.manual_seed(used_seed)

    start = time.time()

    try:
        output = pipe(
            prompt=styled_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height)
        )

        image = output.images[0]
        duration = round(time.time() - start, 2)

        ts = int(time.time())
        path = f"generated_images/img_{ts}_{used_seed}.png"
        image.save(path)

        metadata = {
            "image": image,
            "prompt": prompt,
            "seed": used_seed,
            "time": duration,
            "file": path
        }
        session_images.append(metadata)

        recent = [info["image"] for info in session_images[-10:]]

        return image, recent, f"Generated in {duration}s | Seed: {used_seed}"

    except Exception as e:
        return None, [], f"Error: {e}"

with gr.Blocks(title="Image Generator") as app:
    gr.Markdown("# Text-to-Image Generator")
    gr.Markdown("Generate images using Stable Diffusion models optimized for CPU Compability")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=list(SD_MODELS.keys()),
                    value="Stable Diffusion 1.5",
                    label="Model"
                )

                style_choice = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="None",
                    label="Style"
                )

                with gr.Accordion("Settings", open=False):
                    steps = gr.Slider(10, 50, 20, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 15.0, 7.5, step=0.1, label="Guidance")

                    with gr.Row():
                        width = gr.Slider(256, 768, 512, step=64, label="Width")
                        height = gr.Slider(256, 768, 512, step=64, label="Height")

                    with gr.Row():
                        seed = gr.Number(42, label="Seed")
                        randomize = gr.Checkbox(True, label="Random Seed")

                prompt = gr.Textbox(label="Prompt", lines=3)
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=2, value="low quality, blurry, worst quality, distorted")

                generate = gr.Button("Generate")
                status = gr.Textbox(label="Status")

            with gr.Column(scale=1):
                output_img = gr.Image(label="Result", type="pil")

                with gr.Accordion("History", open=True):
                    history = gr.Gallery(label="Last 10 Images", show_label=True, elem_id="gallery")

    with gr.Tab("Guide"):
        gr.Markdown("""
        ## Prompt Tips
        - Use style, setting, subject, lighting
        - Add details for more refined results

        ## Seed
        - Use fixed seed for repeatability
        - Random seed for creative variance

        ## Performance
        - 512x512 is optimal for CPU
        """)

    generate.click(
        fn=generate_image,
        inputs=[prompt, neg_prompt, model_choice, style_choice, steps, guidance, width, height, seed, randomize],
        outputs=[output_img, history, status]
    )

    app.load(fn=lambda: load_model("Stable Diffusion 1.5"))

app.launch(share=False)
