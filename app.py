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



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists("generated_images"):
    os.makedirs("generated_images")

pipe = None
current_model = "runwayml/stable-diffusion-v1-5"
img_history = []  # keep track of recent gens
last_used = time.time()
IDLE_TIMEOUT = 1800  # 30 mins instead of 10

# models that will work decent on CPU
MODELS = {
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
    "SD 2.1": "stabilityai/stable-diffusion-2-1"
}

# quick prompt boosters
STYLES = {
    "none": "",
    "photo": ", photorealistic, 8k, detailed",
    "art": ", digital art, artstation", 
    "anime": ", anime style, vibrant",
    "painting": ", oil painting, detailed brushwork",
    "sketch": ", pencil sketch, artistic"  # added this one
}

def ping():
    global last_used
    last_used = time.time()

def check_idle():
    global last_used
    if time.time() - last_used > IDLE_TIMEOUT and pipe:
        print("unloading model - been idle too long")
        unload_model()
    Timer(120, check_idle).start()  
def load_model_progress(model_name):
    """generator for UI updates"""
    yield "loading model..."
    result = load_model(model_name)
    yield result

def load_model(model_name):
    global pipe, current_model
    ping()
    
    model_path = MODELS[model_name]
    
    if pipe and current_model == model_path:
        return f"✅ {model_name} already loaded"
    
    try:
        print(f"loading {model_name}...")
        
        if pipe:
            unload_model()
        
        # this usually takes 30-60 seconds on cpu
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            safety_checker=None,  # saves memory
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        # faster scheduler 
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cpu")
        
        # this helps with memory spikes
        pipe.enable_attention_slicing(slice_size="auto")
        
        current_model = model_path
        gc.collect()
        
        print(f"loaded {model_name} successfully")
        return f"✅ {model_name} ready"
        
    except Exception as e:
        print(f"failed to load model: {e}")
        return f"❌ failed: {str(e)}"

def unload_model():
    global pipe
    if pipe:
        print("freeing model memory...")
        del pipe
        pipe = None
        gc.collect()
        return "✅ model unloaded"
    return "no model loaded"

def cleanup_imgs(keep=50):
    """don't let saved images fill up disk"""
    try:
        files = [f for f in os.listdir("generated_images") if f.endswith('.png')]
        if len(files) > keep:
            # sort by date, remove oldest
            files.sort(key=lambda x: os.path.getmtime(f"generated_images/{x}"))
            for old in files[:-keep]:
                os.remove(f"generated_images/{old}")
            print(f"cleaned up {len(files) - keep} old images")
    except:
        pass  # not critical if this fails

def generate_safe(prompt, neg_prompt, model, style, steps, cfg, w, h, seed, random_seed, timeout=600):
    """wrap generation in timeout to prevent hangs - 10 min timeout"""
    result = [None, [], "timed out"]
    
    def run():
        nonlocal result
        result = generate(prompt, neg_prompt, model, style, steps, cfg, w, h, seed, random_seed)
    
    t = threading.Thread(target=run)
    t.daemon = True  # dies with main thread
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        print(f"generation timed out after {timeout}s")
        return None, [], "⚠️ timed out - try simpler settings or restart app"
    
    return result

def generate(prompt, neg_prompt, model_name, style, steps, cfg, width, height, seed, use_random):
    global pipe, img_history
    ping()
    
    if not pipe or MODELS[model_name] != current_model:
        status = load_model(model_name)
        if "❌" in status:
            return None, [], status

    # add style if selected
    full_prompt = prompt
    if style != "none":
        full_prompt += STYLES[style]

    # handle seed
    if use_random:
        seed = random.randint(1, 2**31 - 1)
    else:
        seed = int(seed)
    
    gen = torch.Generator(device="cpu").manual_seed(seed)
    
    # optimized CPU settings for faster generation
    steps = min(int(steps), 20)  # 20 max for speed
    width = min(int(width), 512)
    height = min(int(height), 512)
    
    # force smaller sizes for very slow machines
    if steps > 15:
        width = min(width, 448)
        height = min(height, 448)
    
    start = time.time()
    print(f"generating: {prompt[:30]}... ({steps} steps, {width}x{height})")
    
    try:
        result = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=float(cfg),
            width=width,
            height=height,
            generator=gen
        )
        
        img = result.images[0]
        elapsed = round(time.time() - start, 1)
        
        # save with timestamp + seed for easy tracking
        ts = int(time.time())
        filename = f"generated_images/gen_{ts}_{seed}.png"
        img.save(filename)
        
        cleanup_imgs()
        
        # update history for gallery
        img_data = {
            "image": img,
            "prompt": prompt,
            "seed": seed,
            "time": elapsed,
            "file": filename
        }
        
        img_history.append(img_data)
        if len(img_history) > 6:  # keep last 6
            img_history = img_history[-6:]
        
        gallery_imgs = [item["image"] for item in img_history]
        
        print(f"done in {elapsed}s")
        return img, gallery_imgs, f"✅ generated in {elapsed}s (seed: {seed})"
        
    except Exception as e:
        error = str(e)
        print(f"generation failed: {error}")
        
        # common error handling
        if "cuda" in error.lower() or "meta" in error.lower():
            return None, [], "❌ model issue - try reloading"
        if "memory" in error.lower():
            return None, [], "❌ out of memory - try smaller image"
        
        return None, [], f"❌ error: {error[:50]}..."

# basic batch mode
def batch_gen(prompt, neg_prompt, model, style, steps, cfg, w, h, seed, use_random, count):
    results = []
    
    for i in range(count):
        current_seed = seed + i if not use_random else random.randint(1, 2**31-1)
        
        yield f"batch {i+1}/{count}...", None, []
        
        img, _, status = generate(prompt, neg_prompt, model, style, steps, cfg, w, h, current_seed, False)
        if img:
            results.append(img)
    
    return f"batch done: {len(results)}/{count} success", results, []

# UI setup with Gradio
with gr.Blocks(title="SD Image Generator") as app:
    gr.Markdown("# Stable Diffusion Generator")
    gr.Markdown("*CPU optimized - expect 30-90 seconds per image*")
    
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column():
                model_select = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="SD 1.5",
                    label="Model"
                )
                
                model_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="secondary")
                    unload_btn = gr.Button("Unload", variant="secondary")
                
                style_select = gr.Dropdown(
                    choices=list(STYLES.keys()),
                    value="none",
                    label="Style boost"
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a cat sitting on a windowsill",
                    lines=3
                )
                
                neg_prompt = gr.Textbox(
                    label="Negative prompt",
                    value="blurry, low quality, distorted",
                    lines=2
                )
                
                with gr.Accordion("Settings", open=False):
                    steps = gr.Slider(8, 25, 15, step=1, label="Steps (15 recommended for speed)")
                    cfg = gr.Slider(3.0, 12.0, 7.0, step=0.5, label="CFG Scale")
                    
                    with gr.Row():
                        width = gr.Slider(256, 512, 448, step=64, label="Width (448 for speed)")
                        height = gr.Slider(256, 512, 448, step=64, label="Height")
                    
                    with gr.Row():
                        seed = gr.Number(42, label="Seed")
                        random_seed = gr.Checkbox(True, label="Random seed")
                    
                    batch_count = gr.Slider(1, 3, 1, step=1, label="Batch size (max 3)")
                
                gen_btn = gr.Button("Generate", variant="primary", size="lg")
                status = gr.Textbox(label="Generation status", interactive=False)
            
            with gr.Column():
                output_img = gr.Image(label="Result", type="pil")
                
                with gr.Accordion("Recent", open=True):
                    gallery = gr.Gallery(label="History", show_label=False)
                
                # simple memory monitor
                with gr.Accordion("System", open=False):
                    mem_info = gr.Textbox(label="Memory", interactive=False)
                    refresh_mem = gr.Button("Refresh")
    
    with gr.Tab("Quick Tips"):
        gr.Markdown("""
        ## Speed Tips
        - **15 steps** = good quality, fast (30-60s)
        - **448x448** = faster than 512x512
        - **Lower CFG** (5-7) = faster generation
        
        ## Good Prompts
        - "a red car on a mountain road, digital art"
        - "portrait of a woman, oil painting style"
        - "sunset over ocean, photorealistic"
        
        ## If it's slow
        - Try 10-12 steps max
        - Use 384x384 or smaller
        - Close other programs
        - Restart if it gets stuck
        """)

    # memory checker
    def get_memory():
        try:
            import psutil
            mem = psutil.virtual_memory()
            return f"{mem.percent:.1f}% used, {mem.available/1024**3:.1f}GB free"
        except:
            return "install psutil for memory info"

    # connecting the UI
    load_btn.click(load_model_progress, [model_select], [model_status])
    unload_btn.click(unload_model, [], [model_status])
    refresh_mem.click(get_memory, [], [mem_info])
    
    def handle_gen(*args):
        prompt, neg_prompt, model, style, steps, cfg, w, h, seed, rand_seed, batch = args
        
        if batch <= 1:
            return generate_safe(prompt, neg_prompt, model, style, steps, cfg, w, h, seed, rand_seed)
        else:
            # batch mode
            gen = batch_gen(prompt, neg_prompt, model, style, steps, cfg, w, h, seed, rand_seed, batch)
            for update in gen:
                if update[1] is not None:  # got results
                    return update
            return None, [], "batch failed"
    
    gen_btn.click(
        handle_gen,
        [prompt, neg_prompt, model_select, style_select, steps, cfg, 
         width, height, seed, random_seed, batch_count],
        [output_img, gallery, status]
    )

# start idle checker - less frequent
Timer(120, check_idle).start()

# auto-load default model when the app starts
def init_model():
    return load_model("SD 1.5"), "App ready - model loading..."

app.load(init_model, [], [model_status, status])

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",  # accessible from other devices
        server_port=7860,
        share=True,
        inbrowser=True,  # auto open the browser
        show_error=True
    )
