import argparse
import os

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, \
    DDIMScheduler, EulerAncestralDiscreteScheduler
from lora_diffusion import tune_lora_scale, patch_pipe


def infer(model_path, lora_path, prompt, output_dir, negative_prompt, num_samples, scale, steps, seed):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16)
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
    #                                beta_schedule="scaled_linear", clip_sample=False,set_alpha_to_one=False)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    #pipe.set_use_memory_efficient_attention_xformers(True)
    patch_pipe(
        pipe,
        lora_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    tune_lora_scale(pipe.unet, 1.00)
    tune_lora_scale(pipe.text_encoder, 1.00)
    g_cuda = torch.Generator(device='cuda')
    g_cuda.manual_seed(int(seed))

    height = 512
    width = 512

    with autocast('cuda'), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=int(num_samples),
            num_inference_steps=int(steps),
            guidance_scale=int(scale),
            generator=g_cuda
        ).images

    for i, img in enumerate(images):
        os.makedirs(output_dir, exist_ok=True)
        img_path = output_dir + '/' + str(i) + '.png'
        img.save(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/xyc/800')
    parser.add_argument('--lora_path', default=None)
    parser.add_argument('--prompt', default='photo of guoguo cat in the forest')
    parser.add_argument('--output_dir', default='output/guoguo')
    parser.add_argument('--negative_prompt', default='photoshop')
    parser.add_argument('--num_samples', default=2)
    parser.add_argument('--scale', default=7.5)
    parser.add_argument('--steps', default=50)
    parser.add_argument('--seed', default=135353)
    args = parser.parse_args()
    infer(args.model_path, args.lora_path, args.prompt, args.output_dir, args.negative_prompt, args.num_samples, args.scale, args.steps, args.seed)