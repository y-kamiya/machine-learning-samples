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
            # "torch_dtype": torch.bfloat16,
        }
        self.pipeline = StableDiffusionPipeline.from_single_file(
        # self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model,
            **pipe_kargs,
        ).to(self.config.device)

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler.set_timesteps(self.config.steps)

        if os.path.isfile(self.config.lora):
            self.pipeline.load_lora_weights(self.config.lora)

        if os.path.isfile(self.config.embeddings):
            self.pipeline.load_textual_inversion(self.config.embeddings, token="<V>")

        self.generator = torch.Generator(self.config.device)
        self.generator.manual_seed(0)

    def run(self):
        image = self.pipeline(
            self.config.prompt,
            negative_prompt=self.config.negative,
            num_inference_steps=20,
        )
        image.images[0].save("lora.png")

    def run_decomposed(self):
        text_input = self.pipeline.tokenizer(
            [self.config.prompt] * self.config.batch_size,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.config.device)

        uncond_input = self.pipeline.tokenizer(
            [self.config.negative] * self.config.batch_size,
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(text_input.input_ids)[0]
            uncond_embeddings = self.pipeline.text_encoder(uncond_input.input_ids)[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (
                self.config.batch_size,
                self.pipeline.unet.config.in_channels,
                self.config.height // 8,
                self.config.width // 8,
            ),
            generator=self.generator,
            device=self.config.device,
        ) * self.pipeline.scheduler.init_noise_sigma

        for t in tqdm(self.pipeline.scheduler.timesteps):
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipeline.scheduler.scale_model_input(latent_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.pipeline.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.pipeline.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(image).save("lora_d.png")


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
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    config = parser.parse_args()

    is_cpu = config.cpu or not torch.cuda.is_available()
    config.device_name = "cpu" if is_cpu else "cuda"
    config.device = torch.device(config.device_name)

    model = DiffusersLora(config)
    model.run_decomposed()
