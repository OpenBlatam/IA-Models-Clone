from __future__ import annotations

import asyncio
import base64
import io
from typing import Any, Dict, List, Optional

import gradio as gr
from gradio.themes import Soft
from PIL import Image

from .. import create_modular_ai, SystemMode


_ai_instance = None


async def _get_ai():
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = await create_modular_ai(system_mode=SystemMode.DEVELOPMENT)
    return _ai_instance


def _b64_to_pil(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


async def llm_generate_fn(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            raise gr.Error("Prompt vacío o inválido.")
        if not (1 <= int(max_new_tokens) <= 1024):
            raise gr.Error("max_new_tokens debe estar entre 1 y 1024.")
        if not (0.0 < float(temperature) <= 2.0):
            raise gr.Error("temperature debe estar en (0.0, 2.0].")
        if not (0.0 < float(top_p) <= 1.0):
            raise gr.Error("top_p debe estar en (0.0, 1.0].")
        ai = await _get_ai()
        resp = await ai.process({
            "_engine": "llm.generate",
            "prompt": prompt.strip(),
            "overrides": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        })
        text = resp.get("text")
        if not isinstance(text, str):
            raise gr.Error("Respuesta inválida del modelo.")
        return text
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Error generando texto: {exc}")


async def diffusion_generate_fn(
    prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    steps: int,
    scheduler: str,
    pipeline: str,
) -> Image.Image:
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            raise gr.Error("Prompt vacío o inválido.")
        width = int(width)
        height = int(height)
        steps = int(steps)
        guidance_scale = float(guidance_scale)
        if not (64 <= width <= 2048) or width % 8 != 0:
            raise gr.Error("Width debe ser múltiplo de 8 entre 64 y 2048.")
        if not (64 <= height <= 2048) or height % 8 != 0:
            raise gr.Error("Height debe ser múltiplo de 8 entre 64 y 2048.")
        if not (1 <= steps <= 100):
            raise gr.Error("Steps debe estar entre 1 y 100.")
        if not (0.0 <= guidance_scale <= 20.0):
            raise gr.Error("Guidance scale debe estar entre 0.0 y 20.0.")
        scheduler = str(scheduler).lower()
        pipeline = str(pipeline).lower()
        allowed_schedulers = {"dpmpp_2m", "euler_a", "euler", "ddim", "pndm", "lms"}
        allowed_pipelines = {"auto", "sd15", "sdxl"}
        if scheduler not in allowed_schedulers:
            raise gr.Error("Scheduler inválido.")
        if pipeline not in allowed_pipelines:
            raise gr.Error("Pipeline inválido.")
        ai = await _get_ai()
        resp = await ai.process({
            "_engine": "diffusion.generate",
            "prompt": prompt.strip(),
            "overrides": {
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": steps,
                "scheduler": scheduler,
                "pipeline": pipeline,
            },
        })
        img_b64 = resp.get("image_base64")
        if not isinstance(img_b64, str) or not img_b64:
            raise gr.Error("Error: imagen no generada.")
        return _b64_to_pil(img_b64)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Error generando imagen: {exc}")


async def seo_meta_fn(title: str, summary: str) -> Dict[str, Any]:
    try:
        if not isinstance(title, str) or not title.strip():
            raise gr.Error("Title vacío o inválido.")
        if not isinstance(summary, str) or not summary.strip():
            raise gr.Error("Summary vacío o inválido.")
        ai = await _get_ai()
        return await ai.process({
            "_engine": "seo.meta",
            "title": title.strip(),
            "summary": summary.strip(),
        })
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Error generando meta: {exc}")


async def brand_train_fn(brand_name: str, samples: str) -> Dict[str, Any]:
    try:
        if not isinstance(brand_name, str) or not brand_name.strip():
            raise gr.Error("Brand name vacío o inválido.")
        if not isinstance(samples, str) or not samples.strip():
            raise gr.Error("Samples vacíos o inválidos.")
        samples_list = [s.strip() for s in samples.split("\n") if s.strip()]
        if len(samples_list) < 2:
            raise gr.Error("Incluye al menos 2 ejemplos para entrenar la voz de marca.")
        ai = await _get_ai()
        return await ai.process({
            "_engine": "brand.train",
            "brand_name": brand_name.strip(),
            "samples": samples_list,
        })
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Error entrenando voz de marca: {exc}")


async def social_post_fn(topic: str, brand_name: str) -> str:
    try:
        if not isinstance(topic, str) or not topic.strip():
            raise gr.Error("Topic vacío o inválido.")
        if not isinstance(brand_name, str) or not brand_name.strip():
            raise gr.Error("Brand name vacío o inválido.")
        ai = await _get_ai()
        resp = await ai.process({
            "_engine": "post.social",
            "topic": topic.strip(),
            "brand_name": brand_name.strip(),
        })
        text = resp.get("text")
        if not isinstance(text, str):
            raise gr.Error("Respuesta inválida al generar post.")
        return text
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Error generando post social: {exc}")


async def system_health_fn() -> Dict[str, Any]:
    try:
        ai = await _get_ai()
        return await ai.health_check()
    except Exception as exc:
        raise gr.Error(f"Health check falló: {exc}")


async def system_stats_fn() -> Dict[str, Any]:
    try:
        ai = await _get_ai()
        return ai.get_unified_stats()
    except Exception as exc:
        raise gr.Error(f"Stats falló: {exc}")


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Blaze AI", theme=Soft()) as demo:
        gr.Markdown("""
        # Blaze AI Demo
        Interfaz unificada para generación de texto, imágenes y utilidades de marketing.
        """)
        with gr.Tab("LLM"):
            prompt = gr.Textbox(label="Prompt", lines=6)
            with gr.Row():
                max_new = gr.Slider(1, 512, value=128, step=1, label="Max new tokens")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            out = gr.Textbox(label="Output")
            with gr.Row():
                btn = gr.Button("Generate", variant="primary")
                clear_llm = gr.Button("Clear")
            btn.click(llm_generate_fn, [prompt, max_new, temperature, top_p], out)
            clear_llm.click(lambda: (""), outputs=[out])
            gr.Examples(
                examples=[
                    ["Write a tweet about AI marketing for small businesses.", 120, 0.8, 0.95],
                    ["Summarize the key trends in digital advertising for 2025.", 160, 0.7, 0.9],
                    ["Generate a product description for a minimalist backpack.", 140, 0.9, 0.92],
                ],
                inputs=[prompt, max_new, temperature, top_p],
                label="Examples",
            )

        with gr.Tab("Diffusion"):
            d_prompt = gr.Textbox(label="Prompt", lines=4)
            with gr.Row():
                d_width = gr.Slider(256, 1024, value=512, step=8, label="Width")
                d_height = gr.Slider(256, 1024, value=512, step=8, label="Height")
            with gr.Row():
                d_guidance = gr.Slider(0.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                d_steps = gr.Slider(5, 75, value=30, step=1, label="Steps")
            with gr.Row():
                d_scheduler = gr.Dropdown(["dpmpp_2m", "euler_a", "euler", "ddim", "pndm", "lms"], value="dpmpp_2m", label="Scheduler")
                d_pipeline = gr.Dropdown(["auto", "sd15", "sdxl"], value="auto", label="Pipeline")
            d_image = gr.Image(label="Image")
            with gr.Row():
                d_btn = gr.Button("Generate Image", variant="primary")
                d_clear = gr.Button("Clear")
            d_btn.click(
                diffusion_generate_fn,
                [d_prompt, d_width, d_height, d_guidance, d_steps, d_scheduler, d_pipeline],
                d_image,
            )
            d_clear.click(lambda: None, outputs=[d_image])
            gr.Examples(
                examples=[
                    ["A futuristic city at sunset, ultra-detailed", 768, 512, 7.0, 28, "dpmpp_2m", "auto"],
                    ["Studio photo of a wooden chair, product shot, 8k", 512, 512, 8.0, 30, "euler_a", "sd15"],
                    ["An astronaut riding a horse on Mars, cinematic", 768, 768, 7.5, 35, "euler", "auto"],
                ],
                inputs=[d_prompt, d_width, d_height, d_guidance, d_steps, d_scheduler, d_pipeline],
                label="Examples",
            )

        with gr.Tab("SEO"):
            s_title = gr.Textbox(label="Title")
            s_summary = gr.Textbox(label="Summary", lines=4)
            s_btn = gr.Button("Generate Meta", variant="primary")
            s_out = gr.JSON(label="Result")
            s_btn.click(seo_meta_fn, [s_title, s_summary], s_out)
            gr.Examples(
                examples=[
                    ["How to grow your startup with AI", "A practical guide to leveraging AI tools for early-stage growth and customer acquisition."],
                    ["Top 10 SEO Tips for 2025", "Learn the latest strategies to rank on Google and boost organic traffic."],
                ],
                inputs=[s_title, s_summary],
                label="Examples",
            )

        with gr.Tab("Brand & Social"):
            b_name = gr.Textbox(label="Brand name", value="MiMarca")
            b_samples = gr.Textbox(label="Brand samples (one per line)", lines=5)
            b_btn = gr.Button("Train Brand Voice", variant="secondary")
            b_out = gr.JSON(label="Profile")
            b_btn.click(brand_train_fn, [b_name, b_samples], b_out)

            sp_topic = gr.Textbox(label="Topic", value="growth marketing")
            sp_btn = gr.Button("Generate Social Post", variant="primary")
            sp_out = gr.Textbox(label="Post")
            sp_btn.click(social_post_fn, [sp_topic, b_name], sp_out)
            gr.Examples(
                examples=[
                    ["MiMarca", "Somos una marca centrada en innovación y comunidad.\nHablamos con claridad y cercanía."],
                    ["TechCo", "Productos simples, potentes y accesibles.\nPensamos en crecimiento sostenible."],
                ],
                inputs=[b_name, b_samples],
                label="Brand Voice Examples",
            )

        with gr.Tab("System"):
            h_btn = gr.Button("Health")
            h_out = gr.JSON(label="Health")
            h_btn.click(system_health_fn, outputs=[h_out])
            s_btn2 = gr.Button("Stats")
            s_out2 = gr.JSON(label="Stats")
            s_btn2.click(system_stats_fn, outputs=[s_out2])

    demo.queue(concurrency_count=2)
    return demo


def launch_app(server_name: str = "0.0.0.0", server_port: int = 7860) -> None:
    demo = build_interface()
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_app()


