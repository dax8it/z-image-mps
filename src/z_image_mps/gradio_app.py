import argparse
import os
import random
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Tuple

import gradio as gr
import torch

from .cli import ASPECT_RATIOS, create_generator, load_pipeline, pick_device


def _coerce_int(value: Optional[int], default: int) -> int:
    try:
        v = int(value)
        return v if v > 0 else default
    except Exception:
        return default


def get_available_loras():
    """Get list of available LoRA directories."""
    loras = ["None"]
    loras_dir = "loras"
    if os.path.exists(loras_dir):
        for item in os.listdir(loras_dir):
            if os.path.isdir(os.path.join(loras_dir, item)):
                loras.append(item)
    return loras


@lru_cache(maxsize=1)
def _cached_pipeline(
    device_choice: str,
    attention_backend: str,
    compile_flag: bool,
    cpu_offload: bool,
    lora_name: str,
    lora_scale: float,
) -> Tuple:
    device, dtype = pick_device(device_choice)
    dummy_args = SimpleNamespace(
        attention_backend=attention_backend,
        compile=compile_flag,
        cpu_offload=cpu_offload,
        lora=lora_name if lora_name != "None" else None,
        lora_scale=lora_scale,
    )
    pipe = load_pipeline(dummy_args, device, dtype)
    return pipe, device, dtype


def generate_image(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    aspect: str,
    height: int,
    width: int,
    seed: str,
    attention_backend: str,
    device_choice: str,
    compile_flag: bool,
    cpu_offload: bool,
    lora_name: str,
    lora_scale: float,
):
    steps = max(1, int(steps))
    guidance = float(guidance)

    # Handle dimensions: aspect ratio presets vs custom
    # NOTE: ASPECT_RATIOS stores (width, height) tuples
    if aspect != "custom":
        w, h = ASPECT_RATIOS.get(aspect, (1024, 1024))
    else:
        h = _coerce_int(height, 1024)
        w = _coerce_int(width, 1024)
        # Ensure dimensions are multiples of 16 for the model
        h = (h // 16) * 16
        w = (w // 16) * 16

    pipe, device, dtype = _cached_pipeline(
        device_choice, attention_backend, compile_flag, cpu_offload, lora_name, lora_scale
    )

    # Handle seed: parse from string to preserve precision for large integers
    # (JavaScript loses precision for integers > 2^53)
    seed_str = str(seed).strip() if seed else ""
    try:
        seed_val = int(seed_str) if seed_str and seed_str != "0" else 0
    except (ValueError, TypeError):
        seed_val = 0

    if seed_val == 0:
        # Generate a truly random seed
        seed_val = random.randint(1, 2**63 - 1)
    seed = seed_val

    generator = create_generator(device, seed)

    lora_info = f", lora={lora_name}, lora_scale={lora_scale}" if lora_name != "None" else ""
    info = (
        f"device={device}, dtype={dtype}, steps={steps}, "
        f"guidance={guidance}, size={w}x{h}, seed={seed}, "
        f"attn={attention_backend}, compile={compile_flag}, "
        f"cpu_offload={cpu_offload}{lora_info}"
    )

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )

    return result.images[0], info


def build_app():
    with gr.Blocks(title="Z-Image Turbo") as demo:
        gr.Markdown("# Z-Image Turbo (MPS/CUDA/CPU)")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="Analog film portrait of a skateboarder, shallow depth of field",
                    lines=3,
                )
                negative = gr.Textbox(label="Negative prompt", value="", lines=2)

                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=20,
                        value=9,
                        step=1,
                    )
                    guidance = gr.Slider(
                        label="Guidance scale (Turbo uses 0.0)",
                        minimum=0.0,
                        maximum=5.0,
                        value=0.0,
                        step=0.1,
                    )

                aspect = gr.Dropdown(
                    label="Aspect ratio (select 'custom' to use manual width/height)",
                    choices=list(ASPECT_RATIOS.keys()) + ["custom"],
                    value="1:1",
                )
                with gr.Row():
                    height = gr.Number(
                        label="Height (px) - only used with 'custom' aspect",
                        value=1024,
                        precision=0,
                        interactive=False,
                    )
                    width = gr.Number(
                        label="Width (px) - only used with 'custom' aspect",
                        value=1024,
                        precision=0,
                        interactive=False,
                    )

                # Make width/height editable only when aspect is "custom"
                def update_dimension_interactivity(aspect_val):
                    is_custom = aspect_val == "custom"
                    return gr.update(interactive=is_custom), gr.update(interactive=is_custom)

                aspect.change(
                    fn=update_dimension_interactivity,
                    inputs=[aspect],
                    outputs=[height, width],
                )

                seed = gr.Textbox(
                    label="Seed (0 or empty = random)",
                    value="",
                    placeholder="Leave empty for random seed",
                )

            with gr.Column(scale=2):
                device_choice = gr.Radio(
                    label="Device",
                    choices=["auto", "mps", "cuda", "cpu"],
                    value="auto",
                    interactive=True,
                )
                attention_backend = gr.Radio(
                    label="Attention backend",
                    choices=["sdpa", "flash2", "flash3"],
                    value="sdpa",
                )
                compile_flag = gr.Checkbox(label="torch.compile DiT (CUDA best)", value=False)
                cpu_offload = gr.Checkbox(label="CPU offload (CUDA only)", value=False)

                lora_name = gr.Dropdown(
                    label="LoRA",
                    choices=get_available_loras(),
                    value="None",
                )
                lora_scale = gr.Slider(
                    label="LoRA Scale",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                )

                run_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            image_out = gr.Image(label="Result")
            info = gr.Textbox(label="Run info", interactive=False)

        # Reference section for settings and dimensions
        with gr.Accordion("� Settings Guide & Dimension Reference", open=False):
            gr.Markdown("""
## ⚙️ Recommended Settings

### Steps
| Steps | Speed | Quality | When to Use |
|-------|-------|---------|-------------|
| **4-6** | ⚡ Fastest | Good | Quick previews, testing prompts |
| **8-10** | 🚀 Fast | Great | **Recommended default** (Turbo model is optimized here) |
| **12-15** | 🐢 Slower | Excellent | Final renders, maximum detail |
| **16-20** | 🐌 Slowest | Diminishing returns | Usually unnecessary for Turbo |

> 💡 **Tip:** Z-Image Turbo is designed for speed. **9 steps** is the sweet spot for quality vs time.

### Guidance Scale
| Value | Effect |
|-------|--------|
| **0.0** | **Recommended for Turbo** — model follows prompt naturally |
| **1.0-2.0** | Slightly stronger prompt adherence |
| **3.0-5.0** | Very strict prompt following (may reduce creativity) |

> 💡 **Tip:** Unlike other models, Z-Image Turbo works best with **guidance = 0.0**

### Seed
- **Empty or 0** → Random seed each generation (for variety)
- **Specific number** → Reproducible results (same prompt + seed = same image)

> 💡 **Tip:** Copy a seed from "Run info" to recreate or iterate on an image you like.

---

## 📐 Dimension Reference

### Preset Aspect Ratios
| Aspect | Width × Height | Use Case |
|--------|----------------|----------|
| **1:1** | 1024 × 1024 | Square, social media posts |
| **16:9** | 1280 × 720 | Landscape, widescreen, YouTube |
| **9:16** | 720 × 1280 | Portrait, mobile, TikTok/Reels |
| **4:3** | 1088 × 816 | Classic landscape, presentations |
| **3:4** | 816 × 1088 | Classic portrait |

### Custom Dimensions (up to 2K)
*All values must be multiples of 16. Select "custom" aspect ratio to use these.*

| Type | Dimensions |
|------|------------|
| **Square** | 1536×1536, 2048×2048 |
| **Landscape 16:9** | 1920×1088, 2048×1152 |
| **Portrait 9:16** | 1088×1920, 1152×2048 |
| **Landscape 3:2** | 1536×1024, 1920×1280 |
| **Portrait 2:3** | 1024×1536, 1280×1920 |

> ⚠️ **VRAM Note:** Higher resolutions need more memory. On 16GB Apple Silicon, 1536×1536 or 1920×1088 are recommended max sizes.

---

## 🎨 Quick Start Recipe
For best results, start with these settings:
- **Steps:** 9
- **Guidance:** 0.0
- **Aspect:** 1:1 or 16:9
- **Seed:** Empty (random)

Then adjust based on your needs!
""")

        run_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative,
                steps,
                guidance,
                aspect,
                height,
                width,
                seed,
                attention_backend,
                device_choice,
                compile_flag,
                cpu_offload,
                lora_name,
                lora_scale,
            ],
            outputs=[image_out, info],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Launch a Gradio demo for Z-Image-Turbo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    args = parser.parse_args()

    demo = build_app()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
