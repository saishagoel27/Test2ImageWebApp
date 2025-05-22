import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import time
import random
import os
import gc
import threading
from threading import Timer
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directory for saved images
os.makedirs("generated_images", exist_ok=True)

# Global variables throughout the code
pipe = None
model_id = "runwayml/stable-diffusion-v1-5"
session_images = []  # Store session history
last_activity_time = time.time()
INACTIVITY_TIMEOUT = 600  # 10 minutes of inactivity before unloading model

# Model options to choose from
SD_MODELS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1"
}

# Preset Styles for prompt enhancement
STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", photorealistic, 8k, detailed, sharp focus",
    "Digital Art": ", digital art, trending on artstation, highly detailed",
    "Anime": ", anime style, vibrant colors, highly detailed",
    "Oil Painting": ", oil painting, masterpiece, detailed brushwork, textured",
    "Watercolor": ", watercolor painting, artistic, soft colors, flowing"
}

def update_activity():
    """Update the last activity timestamp"""
    global last_activity_time
    last_activity_time = time.time()

def check_inactivity():
    """Check for inactivity and unload model if inactive"""
    global last_activity_time
    if time.time() - last_activity_time > INACTIVITY_TIMEOUT:
        if pipe is not None:
            logger.info("Unloading model due to inactivity")
            unload_model()
    # Schedule the next check
    Timer(60, check_inactivity).start()

def load_model_with_progress(model_name):
    """Load model with progress updates for UI"""
    yield "🔄 Starting model load process..."
    status = load_model(model_name)
    yield status

def load_model(model_name):
    """Load the selected Stable Diffusion model with optimizations"""
    global pipe, model_id
    update_activity()

    model_id = SD_MODELS[model_name]
    
    try:
        logger.info(f"Loading {model_name} on CPU...")
        unload_model()  # Unloading previous model first
        
        # Loading model with optimized settings
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,  # Disabling safety checker to save memory
            torch_dtype=torch.float32,  # Using float32 for CPU
            device_map=None,  # Explicitly disable device mapping
            low_cpu_mem_usage=True  # Enable low CPU memory usage
        )
        
        # Using efficient scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Setting device to CPU explicitly for all components
        pipe = pipe.to("cpu")
        
        # Enabling memory optimizations
        pipe.enable_attention_slicing(slice_size="auto")
        
        # model offloading where possible - remove this for CPU-only operation
        # Commenting out CPU offload as it can cause issues on CPU-only systems
        # try:
        #     pipe.enable_model_cpu_offload()
        # except:
        #     logger.info("Model CPU offload not available in this diffusers version")
        
        #  garbage collection
        gc.collect()
        
        logger.info(f"Model {model_name} loaded successfully on CPU!")
        return f"✅ {model_name} loaded successfully on CPU"
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return f"❌ Error loading model: {str(e)}"

def unload_model():
    """Unloading model to free memory"""
    global pipe
    if pipe is not None:
        logger.info("Unloading model from memory")
        del pipe
        pipe = None
        gc.collect()
        # Remove CUDA cache clearing for CPU-only operation
        # try:
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        # except Exception as e:
        #     logger.warning(f"Skipping CUDA cleanup: {e}")
        return "✅ Model unloaded to free memory"
    return "⚠️ No model currently loaded"


def cleanup_old_images(max_files=50):
    """Clean up old generated images to prevent disk space issues"""
    try:
        files = os.listdir("generated_images")
        if len(files) > max_files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join("generated_images", x)))
            for old_file in files[:len(files) - max_files]:
                os.remove(os.path.join("generated_images", old_file))
            logger.info(f"Cleaned up {len(files) - max_files} old image files")
    except Exception as e:
        logger.error(f"Error cleaning up old images: {str(e)}")

def generate_with_timeout(prompt, negative_prompt, model_name, style_preset,
                         steps, guidance, width, height, seed, use_random_seed, timeout=300):
    """Generate image with timeout to prevent hanging"""
    result = [None, [], "Generation timed out"]
    
    def target():
        nonlocal result
        result = generate_image(prompt, negative_prompt, model_name, style_preset,
                                steps, guidance, width, height, seed, use_random_seed)
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        logger.warning(f"Generation timed out after {timeout}s")
        return None, [], f"⚠️ Generation timed out after {timeout}s - try simpler parameters"
    
    return result

def generate_image(prompt, negative_prompt, model_name, style_preset,
                   steps, guidance, width, height, seed, use_random_seed):
    """Generate image with advanced options and optimization"""
    global pipe, session_images
    update_activity()
    
    # Memory usage logging - remove GPU memory logging for CPU-only
    # if torch.cuda.is_available():
    #     mem_before = torch.cuda.memory_allocated() / 1024**2
    #     logger.info(f"GPU Memory before generation: {mem_before:.2f} MB")
    
    # Check if model needs to be loaded or changed
    if pipe is None or SD_MODELS[model_name] != model_id:
        status = load_model(model_name)
        if "❌" in status:
            return None, [], status

    # Apply style preset if selected
    if style_preset != "None":
        enhanced_prompt = prompt + STYLE_PRESETS[style_preset]
    else:
        enhanced_prompt = prompt

    # Set seed
    generator = None
    if use_random_seed:
        used_seed = random.randint(1, 2147483647)
    else:
        used_seed = int(seed)
    
    # Create CPU generator
    generator = torch.Generator(device="cpu").manual_seed(used_seed)

    start_time = time.time()
    logger.info(f"Starting generation with prompt: {prompt[:50]}...")

    try:
        # Generating image with CPU-optimized parameters
        actual_steps = min(int(steps), 30)  # Cap steps at 30 for CPU
        
        # For CPU, adjust the dimensions if they're too large
        actual_width = min(int(width), 512)
        actual_height = min(int(height), 512)
        
        if actual_width != int(width) or actual_height != int(height) or actual_steps != int(steps):
            logger.info(f"Adjusting parameters for CPU: steps={actual_steps}, size={actual_width}x{actual_height}")
        
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=actual_steps,
            guidance_scale=float(guidance),
            width=actual_width,
            height=actual_height,
            generator=generator
        )

        image = result.images[0]
        gen_time = round(time.time() - start_time, 2)
        logger.info(f"Generation completed in {gen_time}s")

        # Saving the image
        timestamp = int(time.time())
        filename = f"generated_images/img_{timestamp}_{used_seed}.png"
        image.save(filename)

        # Clean up old images if they are too many
        cleanup_old_images()

        # Add to the session history with metadata
        image_info = {
            "image": image,
            "prompt": prompt,
            "seed": used_seed,
            "time": gen_time,
            "file": filename
        }
        
        # Limiting session history size for memory efficiency
        session_images.append(image_info)
        if len(session_images) > 5:
            session_images = session_images[-5:]
        
        # Return only recent history
        recent_images = [info["image"] for info in session_images]

        # Log memory usage after generation - remove GPU logging for CPU-only
        # if torch.cuda.is_available():
        #     mem_after = torch.cuda.memory_allocated() / 1024**2
        #     logger.info(f"GPU Memory after generation: {mem_after:.2f} MB")

        return image, recent_images, f"Image generated in {gen_time}s (Seed: {used_seed})"

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Generation error: {error_msg}")
        if "CUDA" in error_msg or "meta" in error_msg.lower():
            return None, [], "❌ Device Error: Model loading failed - try reloading the model"
        elif "attention_mask" in error_msg:
            return None, [], "❌ Model processing error: Try a different prompt"
        return None, [], f"❌ Error: {error_msg}"

def batch_generate(prompt, negative_prompt, model_name, style_preset,
                  steps, guidance, width, height, seed, use_random_seed, count=3):
    """Generate multiple images in batch mode"""
    images = []
    status_messages = []
    
    for i in range(count):
        current_seed = seed + i if not use_random_seed else random.randint(1, 2147483647)
        
        yield f"Generating image {i+1}/{count}...", None, []
        
        image, _, status = generate_image(
            prompt, negative_prompt, model_name, style_preset,
            steps, guidance, width, height, current_seed, False
        )
        
        if image is not None:
            images.append(image)
            status_messages.append(status)
    
    return f"Batch generation complete. {len(images)}/{count} successful.", images, status_messages

# Developing the Gradio interface
with gr.Blocks(title="Advanced Text-to-Image Generator") as demo:
    gr.Markdown("# Advanced Text-to-Image Generator")
    gr.Markdown("Generate high-quality images from text descriptions with CPU-optimized settings.")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection and parameters
                model_dropdown = gr.Dropdown(
                    choices=list(SD_MODELS.keys()),
                    value="Stable Diffusion 1.5",
                    label="Select Model"
                )
                
                model_status = gr.Textbox(label="Model Status")
                load_btn = gr.Button("Load/Change Model")
                unload_btn = gr.Button("Unload Model (Free Memory)")

                style_dropdown = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="None",
                    label="Style Preset"
                )

                with gr.Accordion("Generation Settings", open=False):
                    steps_slider = gr.Slider(10, 50, 20, step=1, label="Inference Steps (CPU: max 30)")
                    guidance_slider = gr.Slider(1.0, 15.0, 7.5, step=0.1, label="Guidance Scale")

                    with gr.Row():
                        width_slider = gr.Slider(256, 768, 512, step=64, label="Width (CPU: max 512)")
                        height_slider = gr.Slider(256, 768, 512, step=64, label="Height (CPU: max 512)")

                    with gr.Row():
                        seed_number = gr.Number(42, label="Seed")
                        random_seed = gr.Checkbox(True, label="Use Random Seed")
                        
                    with gr.Row():
                        batch_size = gr.Slider(1, 5, 1, step=1, label="Batch Size (1 = single image)")

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate",
                    lines=3
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt (what to avoid)",
                    placeholder="Low quality, blurry, distorted face",
                    lines=2,
                    value="low quality, blurry, worst quality, distorted"
                )

                generate_btn = gr.Button("Generate Image", variant="primary")
                status_output = gr.Textbox(label="Status")

            with gr.Column(scale=1):
                # Result display and gallery
                image_output = gr.Image(label="Generated Image", type="pil")

                with gr.Accordion("Generation History", open=True):
                    gallery = gr.Gallery(
                        label="Recent Generations",
                        show_label=True,
                        elem_id="gallery"
                    )
                    
                with gr.Accordion("System Monitor", open=False):
                    mem_usage = gr.Textbox(label="Memory Usage")
                    refresh_btn = gr.Button("Refresh System Stats")

    with gr.Tab("Help & Tips"):
        gr.Markdown("""
        ## Tips for Better Results
        ### Effective Prompts
        - Be specific and detailed in your descriptions
        - Mention lighting, style, camera angle if relevant
        - Use artistic references ("in the style of...")
        ### Negative Prompts
        Use negative prompts to avoid common issues:
        - "low quality, blurry, worst quality" - Avoid poor quality
        - "distorted face, bad anatomy" - Fix common anatomy issues
        - "text, watermark" - Avoid text in images
        ### CPU Optimization Tips
        - For CPU mode: 512x512 or smaller is recommended
        - Lower steps (20-30) work best for CPU generation
        - Expect 1-2 minutes per image on CPU
        - Unload model when not in use to free memory
        ### Seed Control
        - Uncheck "Random Seed" and use the same seed to create variations with different prompts
        - Save seeds of images you like to recreate similar compositions
        """)
        
    with gr.Tab("About"):
        gr.Markdown("""
        ## Advanced Text-to-Image Generator
        This application runs Stable Diffusion entirely on CPU with optimizations for memory usage and performance.
        
        ### Technical Details
        - Models: Stable Diffusion 1.5 and 2.1
        - Scheduler: DPMSolverMultistepScheduler (faster than default)
        - Memory optimization: Attention slicing, model offloading
        - Automatic timeout prevention and resource management
        
        ### System Requirements
        - CPU: Minimum 4 cores recommended
        - RAM: Minimum 8GB, 16GB recommended
        - Generation Time: 1-3 minutes per image depending on settings
        """)

    # Defining update function for system monitor
    def update_system_stats():
        mem_info = "CPU Memory: Not available"
        # Remove GPU memory monitoring for CPU-only operation
        # if torch.cuda.is_available():
        #     mem_allocated = torch.cuda.memory_allocated() / 1024**2
        #     mem_reserved = torch.cuda.memory_reserved() / 1024**2
        #     mem_info = f"GPU Memory: {mem_allocated:.2f}MB allocated, {mem_reserved:.2f}MB reserved"
        # else:
        try:
            import psutil
            vm = psutil.virtual_memory()
            mem_info = f"System Memory: {vm.percent}% used, {vm.available / 1024**3:.2f}GB available"
        except ImportError:
            mem_info = "System Memory: psutil not available - install with 'pip install psutil'"
        except Exception as e:
            mem_info = f"System Memory: Error reading stats - {str(e)}"
        return mem_info

    # Setting up event handlers
    load_btn.click(
        fn=load_model_with_progress,
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    unload_btn.click(
        fn=unload_model,
        inputs=[],
        outputs=[model_status]
    )
    
    refresh_btn.click(
        fn=update_system_stats,
        inputs=[],
        outputs=[mem_usage]
    )

    def handle_generation(prompt, negative_prompt, model_name, style_preset,
                        steps, guidance, width, height, seed, use_random_seed, batch_count):
        if batch_count <= 1:
            # Single image generation
            return generate_with_timeout(
                prompt, negative_prompt, model_name, style_preset,
                steps, guidance, width, height, seed, use_random_seed
            )
        else:
            # Batch generation
            generator = batch_generate(
                prompt, negative_prompt, model_name, style_preset,
                steps, guidance, width, height, seed, use_random_seed, batch_count
            )
            # Process the generator for batch mode
            for status_update, img, imgs in generator:
                if img is not None:
                    return img, imgs, status_update
            return None, [], "Batch generation failed"

    generate_btn.click(
        fn=handle_generation,
        inputs=[
            prompt_input,
            negative_prompt,
            model_dropdown,
            style_dropdown,
            steps_slider,
            guidance_slider,
            width_slider,
            height_slider,
            seed_number,
            random_seed,
            batch_size
        ],
        outputs=[image_output, gallery, status_output]
    )

    # Starting the inactivity checker
    Timer(60, check_inactivity).start()
    
    # will Load the default model at start
    demo.load(fn=lambda: load_model("Stable Diffusion 1.5"))

# Launch the app
if __name__ == "__main__":
    demo.launch()
