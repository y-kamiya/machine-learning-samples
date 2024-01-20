import argparse
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from tqdm.auto import tqdm


class DiffusersSample:
    def __init__(self, config):
        self.config = config
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="text_encoder",
            use_safetensors=True,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
        )
        self.generator = torch.Generator(self.config.device)
        self.generator.manual_seed(0)

        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        self.scheduler.set_timesteps(self.config.steps)

        self.vae.to(self.config.device)
        self.text_encoder.to(self.config.device)
        self.unet.to(self.config.device)

    def run(self):
        text_input = self.tokenizer(
            [self.config.prompt] * self.config.batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.config.device)
            )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * self.config.batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.config.device))[
            0
        ]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (
                self.config.batch_size,
                self.unet.config.in_channels,
                self.config.height // 8,
                self.config.width // 8,
            ),
            generator=self.generator,
            device=self.config.device,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(image).save("image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
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

    model = DiffusersSample(config)
    model.run()
