from cog import BasePredictor, Input, Path
import re
import torch
from PIL import Image
from compel import Compel
from clip_interrogator import Config, Interrogator
from diffusers import StableDiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler

MODEL_NAME = "SG161222/Realistic_Vision_V5.1_noVAE"
MODEL_CACHE = "checkpoints"
VAE_CACHE = "vae-cache"


def filter_age_and_gender(prompt_str: str):
    # replace male/female/man/woman with human
    gender_replacements = ["male", "female", "man", "woman", "boy", "girl"]
    for word in gender_replacements:
        prompt_str = re.sub(r'\b' + word + r'\b', "human", prompt_str)
    # remove sections referring to age
    prompt_str = re.sub(r',\s*[\d\s\-]*year[s]?[\s\-]*old', "", prompt_str)
    # remove sections between commas
    prompt_str = re.sub(r',\s*[^,]*hair[^,]*,', ",", prompt_str)
    return prompt_str


class Predictor(BasePredictor):
    def setup(self):
        # Load RV5.1
        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
            cache_dir=VAE_CACHE
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            vae=vae,
            use_safetensors=True,
            cache_dir=MODEL_CACHE,
        ).to("cuda")
        # Load Clip Interrogator
        self.ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        # Load Compel
        self.compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Man's image"),
        image2: Path = Input(description="Woman's image"),
        gender: str = Input(
            default="boy",
            choices=["boy", "girl"],
            description="Choose gender",
        ),
        steps: int = Input(description="Number of inference steps", ge=0, le=100, default=25),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=728),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", default=None),
    ) -> Path:
        # Set random seed
        if seed is None:
            seed = torch.randint(0, 2147483647, (1,)).item()
        print(f"Seed is: {seed}")
        generator = torch.manual_seed(seed)

        img1 = Image.open(image).convert('RGB')
        img2 = Image.open(image2).convert('RGB')
        prompt1 = self.ci.interrogate_fast(img1)
        prompt2 = self.ci.interrogate_fast(img2)
        prompt1 = filter_age_and_gender(prompt1)
        prompt2 = filter_age_and_gender(prompt2)

        prompt_embeds = self.compel_proc(
            f'("RAW photo, a close up portrait of a toddler {gender}, natural skin, 8k uhd, high quality, film grain, Fujifilm XT3,high quality portrait", looking at camera, "{prompt1}", "{prompt2}").blend(1.6, 1.0, 1.0)')
        negative_embeds = self.compel_proc(
            "(old, adult, bags under eyes, deformed iris, deformed pupils, cgi, render, 3d, sketch, cartoon, drawing, anime, facial hair:1.4), adult, chubby, hat, earing, bling, chains, jewelry, text, close up, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers")
      
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        image = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=steps,
            width=width,
            height=height
        ).images[0]

        output_path = "/tmp/output.png"
        image.save(output_path)

        return Path(output_path)
