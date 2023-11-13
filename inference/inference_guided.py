import sys
sys.path.append("./kornia_canny")

from kornia_canny import *

import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import cv2

sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPProcessor, CLIPModel

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--outdir",
            type=str,
            default="./cv_test"
        )
    parser.add_argument(
            "--emb",
            type=str,
            default="./InST/logs/etc/checkpoints/embeddings.pt"
        )
    parser.add_argument(
            "--content",
            type=str,
            default="./data/content/8/8.png"
        )
    parser.add_argument(
            "--wt",
            type=float,
            default=0.4,
            help="path to specific_step"
        )
    return parser

#cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def resize_prev(img, basewidth):
    basewidth = basewidth
    wpercent = (basewidth/float(img.size()[1]))
    wsize = int((float(img.size()[0])*float(wpercent)))
    img = img.resize((wsize, basewidth), Image.LANCZOS)
    
    return img

#config 정의
config='./InST/configs/stable-diffusion/v1-inference.yaml'
ckpt="./InST/models/sd/sd-v1-4.ckpt"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")


#(case 1). original InST inference (without canny)
def main(outdir=None, emb=None, prompt = '', content_dir = '', style_dir='',ddim_steps = 50,strength = 0.5, model = model, seed=42):
    count = 'love_test_10' #self
    
    ddim_eta=0.0
    n_iter=1
    C=4
    f=8
    n_samples=1
    n_rows=0
    scale=10.0 #10
    
    precision="autocast"
    seed_everything(seed)
    
    model.embedding_manager.load(emb)
    model = model.to(device)

    outdir = os.path.join(outdir, 'style_trans')
    os.makedirs(outdir, exist_ok=True)
    
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    base_count = len(os.listdir(outpath))
    grid_count = len(os.listdir(outpath)) + 10
    
    
    style_image = load_img(style_dir).to(device)
    style_image = repeat(style_image, '1 ... -> b ...', b=batch_size)
    
    content_name =  content_dir.split('/')[-1].split('.')[0]
    content_image = load_img(content_dir).to(device)
    content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
    content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

    init_latent = content_latent

    #sampler 함수 시작
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""], style_image)

                        c = model.get_learned_conditioning(prompts, style_image)
                        
                        # stochastic inversion
                        t_enc = int(strength * 1000) #500
                        
                        x_noisy = model.q_sample(x_start=content_latent, t=torch.tensor([t_enc]*batch_size).to(device)) #1.forward
                        model_output = model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c) #2.pred_noise -> #3.reverse
                        z_enc = sampler.stochastic_encode(content_latent, torch.tensor([t_enc]*batch_size).to(device),
                                                          noise = model_output, use_original_steps = True)
                        
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
                save_dir = os.path.join(outpath, count+'-'+content_name+'-'+prompt+f'-{grid_count:04}'+'-'+f'{strength}'+'.png')
                output.save(save_dir)
                grid_count += 1

                toc = time.time()
    return output, save_dir, count


#(case 2). with canny 
def canny_test_sca(outdir=None, emb=None, prompt = '', content_dir = '', style_dir='',ddim_steps = 50, strength = 0.5, model = model, seed=42, custom=50):
    count = 'love_test_10' #self
    
    ddim_eta=0.0
    n_iter=1
    C=4
    f=8
    n_samples=1
    n_rows=0
    scale=10.0 #10
    
    precision="autocast"
    cannydir = outdir
    seed_everything(seed)
    
    model.embedding_manager.load(emb)
    model = model.to(device)

    outdir = os.path.join(outdir, 'style_trans')
    os.makedirs(outdir, exist_ok=True)
    #styled canny path
    styled_canny_path = os.path.join(outdir, 'style_canny')
    os.makedirs(styled_canny_path, exist_ok=True)
    
    outpath = outdir
    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    base_count = len(os.listdir(outpath))
    grid_count = len(os.listdir(outpath)) + 10
    
    style_image = load_img(style_dir).to(device)
    style_image = repeat(style_image, '1 ... -> b ...', b=batch_size)
    style_latent = model.get_first_stage_encoding(model.encode_first_stage(style_image))  # move to latent space

    content_name =  content_dir.split('/')[-1].split('.')[0]
    content_image = load_img(content_dir).to(device)
    content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
    content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

    #canny edge image 추가
    canny_path = save_canny(content_dir, cannydir) #path 
    canny_image = load_img(canny_path).to(device)
    canny_image = repeat(canny_image, '1 ... -> b ...', b=batch_size)
    canny_latent = model.get_first_stage_encoding(model.encode_first_stage(canny_image))

    init_latent = content_latent

    #sampler 함수 시작
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    

    #1). Canny styling
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""], style_image)
                        
                        #conditioning data
                        c = model.get_learned_conditioning(prompts, style_image)
                        ca = model.get_learned_conditioning(prompts, canny_image)

                        t_enc = int(strength * 1000) #500
                        
                        #noising start
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc]*batch_size).to(device)) #1.forward

                        if custom == 0:
                            model_output = model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c) #2.pred_noise -> #3.reverse
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device),
                                                            noise = model_output, use_original_steps = True)
                        else:
                            #1. canny conditioning
                            model_output = model.apply_model(x_noisy, torch.tensor([custom]*batch_size).to(device), ca) #2.pred_noise -> #3.reverse
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([custom]*batch_size).to(device),
                                                            noise = model_output, use_original_steps = True)
                            #2. style conditioning
                            model_output = model.apply_model(z_enc, torch.tensor([t_enc]*batch_size).to(device), c) #2.pred_noise -> #3.reverse
                            z_enc = sampler.stochastic_encode(canny_latent, torch.tensor([t_enc]*batch_size).to(device),
                                                            noise = model_output, use_original_steps = True)
                            
                        #decoding
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                scanny_output = Image.fromarray(grid.astype(np.uint8))
                idx_num = len(os.listdir(styled_canny_path))
                sc_path = os.path.join(styled_canny_path, 'canny-'+f'{idx_num:04}'+'-'+f'{strength}'+'.jpeg')
                scanny_output.save(sc_path)
                grid_count += 1

                toc = time.time()
                
    #2). styled canny image load
    scanny_path=sc_path
    scanny_image = load_img(scanny_path).to(device)
    scanny_image = repeat(scanny_image, '1 ... -> b ...', b=batch_size)
    scanny_latent = model.get_first_stage_encoding(model.encode_first_stage(scanny_image))
    
    #3. canny conditioned image generation
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""], style_image)
                        
                        #conditioning data
                        c = model.get_learned_conditioning(prompts, style_image)
                        sca = model.get_learned_conditioning(prompts, scanny_image)
                        
                        t_enc = int(strength * 1000) #500
                        
                        #noising start
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc]*batch_size).to(device)) 

                        if custom == 0:
                            model_output = model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c) 
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device),
                                                            noise = model_output, use_original_steps = True)
                        else:
                            #(1). style canny conditioning
                            model_output = model.apply_model(x_noisy, torch.tensor([1]*batch_size).to(device), sca) 
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([1]*batch_size).to(device),
                                                            noise = model_output, use_original_steps = True)
                            #(2). style conditioning
                            model_output = model.apply_model(z_enc, torch.tensor([t_enc]*batch_size).to(device), c) 
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device),
                                                            noise = model_output, use_original_steps = True)
                            
                            # cf. styled canny latent로 하는 경우
                            # z_enc = sampler.stochastic_encode(scanny_latent, torch.tensor([t_enc]*batch_size).to(device),
                            #                                 noise = model_output, use_original_steps = True)
                        
                        #decoding
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            #base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid2 = torch.stack(all_samples, 0)
                grid2 = rearrange(grid2, 'n b c h w -> (n b) c h w')
                grid2 = make_grid(grid2, nrow=n_rows)

                # to image
                grid2 = 255. * rearrange(grid2, 'c h w -> h w c').cpu().numpy()
                output2 = Image.fromarray(grid2.astype(np.uint8))
                idx_num = len(os.listdir(outpath))-1
                save_dir = os.path.join(outpath, 'style_trans-'+f'{idx_num:04}'+'-'+f'{strength}'+'.png')
                
                output2 = output2.resize((300, 300), resample=PIL.Image.LANCZOS)
                output2.save(save_dir)

                base_count += 1

                toc = time.time()

    return output2, save_dir, count


def style_start_guide(content_dir, style_dir, weight, emb_path, outdir=None, ddim_steps=50, custom=50):
    return canny_test_sca(prompt = '*', \
                content_dir = content_dir, \
                style_dir = style_dir, \
                ddim_steps = ddim_steps, \
                strength = weight, \
                seed=42, \
                model = model,
                emb=emb_path,
                outdir=outdir,
                custom=custom)


def style_start_org(content_dir, style_dir, weight, emb_path, outdir=None, ddim_steps=50):
    return main(prompt = '*', \
                content_dir = content_dir, \
                style_dir = style_dir, \
                ddim_steps = ddim_steps, \
                strength = weight, \
                seed=42, \
                model = model,
                emb=emb_path,
                outdir=outdir)



if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()

    style_start_guide('./data/content/8/8.png', 
                './data/style_1/1.png',
                0.4,
                './InST/logs/ugly/checkpoints/embeddings.pt',
                './cv_test/inf_Test')
    
    #original InST inference
    # main(prompt = '*', \
    #     content_dir = './data/content/8/8.png', \
    #     style_dir = './data/style_1/1.png', \
    #     ddim_steps = 50, \
    #     strength = args.wt, \
    #     seed=42, \
    #     model = model, \
    #     emb = args.emb, \
    #     outdir = './cv_test/inf_Test')
