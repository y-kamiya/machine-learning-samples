import argparse
import torch
import gradio as gr
from diffusers import DiffusionPipeline


class Pipeline:
    def __init__(self, config):
        self.config = config

        self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipeline.to(self.config.device)

    def text2img(self, prompt, negative_prompt, width, height, n_steps, guidance_scale):
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
        ).images[0]


def main(config):
    pipeline = Pipeline(config)

    app = gr.Interface(
        fn=pipeline.text2img,
        inputs=[
            gr.TextArea(label="Prompt"),
            gr.TextArea(label="Negative Prompt"),
            gr.Number(label="Width", value=512),
            gr.Number(label="Height", value=512),
            gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling Step"),
            gr.Slider(minimum=1, maximum=100, value=7.5, step=0.5, label="CFG"),
        ],
        outputs=gr.Image(height=512, width=512),
    )

    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    config = parser.parse_args()

    is_cpu = config.cpu or not torch.cuda.is_available()
    config.device_name = "cpu" if is_cpu else "cuda"
    config.device = torch.device(config.device_name)

    main(config)
