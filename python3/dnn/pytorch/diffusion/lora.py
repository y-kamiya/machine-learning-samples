import os
import argparse
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from tqdm.auto import tqdm


class DiffusersLora:
    def __init__(self, config):
        self.config = config

        pipe_kargs = {
            "use_safetensors": True,
            "load_safety_checker": False,
            "torch_dtype": torch.bfloat16,
        }
        self.pipeline = StableDiffusionPipeline.from_single_file(
        # self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model,
            **pipe_kargs,
        ).to(self.config.device)

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)

        if os.path.isfile(self.config.lora):
            self.pipeline.load_lora_weights(self.config.lora)

        if os.path.isfile(self.config.embeddings):
            self.pipeline.load_textual_inversion(self.config.embeddings)

    def run(self):
        image = self.pipeline(
            self.config.prompt,
            negative_prompt="easynegative",
            num_inference_steps=20,
        )
        image.images[0].save("lora.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora", default="")
    parser.add_argument("--embeddings", default="")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="size of batch")
    parser.add_argument(
        "--prompt", default="a photograph of an astronaut riding a horse"
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    config = parser.parse_args()

    is_cpu = config.cpu or not torch.cuda.is_available()
    config.device_name = "cpu" if is_cpu else "cuda"
    config.device = torch.device(config.device_name)

    model = DiffusersLora(config)
    model.run()
