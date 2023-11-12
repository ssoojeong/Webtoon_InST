import sys
sys.path.append('./DIS/IS_Net')
sys.path.append('./InST')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from transformers import pipeline
import gradio as gr
import torch
import os

#배경 분리
from DIS.IS_Net.Inference import *
from DIS.IS_Net.models.isnet import ISNetGTEncoder, ISNetDIS

#inference code
from inference.total_inference_demo import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. with canny
def process_canny(input_image, image_resolution, ddim_steps, fore_scale, custom, embtype):
    emb = None
    style_file = None
    outdir = './demo_output/yeosin'
    if embtype=='Type 1':
        emb = './logs/yeosin/checkpoints/embeddings.pt'
        style_file = './data/yeosin/yeosin.jpg'
    
    content_dir = input_image
    fg_wt = fore_scale
    #bg_wt = back_scale
    name = 'original'
    ddim_step = ddim_steps
    dst3 = image_to_image_canny(outdir, emb, style_file, content_dir, fg_wt, name, ddim_step, custom=custom)
    
    return [dst3]

def create_demo_canny(max_images=12, default_num_images=3, theme=None):
    with gr.Blocks(theme=theme) as demo:
        with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type='filepath', shape=(100, 100))
                    run_button = gr.Button(label="Run")
                    
                                                
                    with gr.Accordion("Advanced options", open=True):
                        #num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=520, step=64)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=200, value=20, step=1)
                        fore_scale = gr.Slider(label="Style Guidance Scale", minimum=0.1, maximum=1.0, value=0.8, step=0.1)
                        canny_steps = gr.Slider(label="Canny Guidance Steps", minimum=0, maximum=100, value=1, step=1)
                        #back_scale = 0.4
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=1, height=635)
                    
        with gr.Accordion("Style Image Type", open=True, label="Style Image Type"):
                        click = gr.Radio(label="Target Style Type", choices=["Type 1"], value='Type 1')
                        with gr.Row():
                            #emb_type = gr.Radio(label="Style Image type", choices=["Woman", "Man"], value="Woman")
                            emb_type = gr.Image(label="Type 1", value="./data/ui/3.jpg", type='filepath', shape=(1,1))
        gr.Examples(examples=[['./data/face/0048132.png', 512, 100, 0.7, 50, 'Type 1'],
                              ["./data/face/0048145.png", 512, 200, 0.6, 50, 'Type 1']],
                    inputs=[input_image, image_resolution, ddim_steps, fore_scale, canny_steps, click],
                    fn=process_canny,
                    outputs=result_gallery,
                    cache_examples=True
                   )
        
        ips = [input_image, image_resolution, ddim_steps, fore_scale, canny_steps, click]
        run_button.click(fn=process_canny, inputs=ips, outputs=[result_gallery])
    return demo

#2. without canny
def process(input_image, image_resolution, ddim_steps, fore_scale, embtype):
    emb = None
    style_file = None
    outdir = './demo_output/yeosin'
    
    if embtype=='Type 1':
        emb = './logs/yeosin/checkpoints/embeddings.pt'
        style_file = './data/yeosin/yeosin.jpg'
        
    content_dir = input_image
    fg_wt = fore_scale
    #bg_wt = back_scale #배경 분리 실시할 경우
    name = 'original'
    ddim_step = ddim_steps
    dst3 = image_to_image_org(outdir, emb, style_file, content_dir, fg_wt, name, ddim_step)
    
    #img = Image.open(dst3_path)
    return [dst3]


def create_demo_org(max_images=12, default_num_images=3, theme=None):
    with gr.Blocks(theme=theme) as demo:
        with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type='filepath', shape=(100, 100))
                    run_button = gr.Button(label="Run")
                    
                                                
                    with gr.Accordion("Advanced options", open=True):
                        #num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=520, step=64)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=200, value=200, step=1)
                        fore_scale = gr.Slider(label="Style Guidance Scale", minimum=0.1, maximum=1.0, value=0.7, step=0.1)
                        #canny_steps = gr.Slider(label="Canny Guidance Steps", minimum=1, maximum=100, value=50, step=1)
                        #back_scale = 0.4
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=1, height=512)
        
        with gr.Accordion("Style Image Type", open=True, label="Style Image Type"):
                        click = gr.Radio(label="Target Style Type", choices=["Type 1"], value='Type 1')
                        with gr.Row():
                            emb_type = gr.Image(label="Type 1", value="./data/ui/3.jpg", type='filepath', shape=(1,1))
        gr.Examples(examples=[['./data/face/0048132.png', 512, 100, 0.7, 'Type 1'],
                              ["./data/face/0048145.png", 512, 200, 0.6, 'Type 1']],
                    inputs=[input_image, image_resolution, ddim_steps, fore_scale, click],
                    fn=process,
                    outputs=result_gallery,
                    cache_examples=True
                   )
        
        ips = [input_image, image_resolution, ddim_steps, fore_scale, click]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    return demo


