import sys
sys.path.append("/userHome/userhome1/sojeong/demo/")
sys.path.append('/userHome/userhome1/sojeong/demo/DIS/IS_Net') #배경 분리
sys.path.append('/userHome/userhome1/sojeong/demo/InST')

import gradio as gr
import os

#demo app load
from app_yeosin import create_demo_canny as create_demo_yeosin
from app_love import create_demo_canny as create_demo_love
from app_ugly import create_demo_canny as create_demo_ugly
from app_etc import create_demo_canny as create_demo_etc

#default
MAX_IMAGES = int(os.getenv('MAX_IMAGES', '3'))
DEFAULT_NUM_IMAGES = min(MAX_IMAGES, int(os.getenv('DEFAULT_NUM_IMAGES', '1')))

#색상
theme = gr.themes.Default().set(
    button_primary_background_fill="#FF0000",
    button_primary_background_fill_dark="#AAAAAA",
)

#demo block 형성
block = gr.Blocks(theme=theme).queue()
with block:
    with gr.Row():
        gr.Markdown("## 너의 웹툰 캐릭터는?")
    with gr.Tabs():
        with gr.TabItem('여신강림'):
            create_demo_yeosin(
                                max_images=MAX_IMAGES,
                                default_num_images=DEFAULT_NUM_IMAGES,
                                theme=theme)
        with gr.TabItem('어글리'):
            create_demo_ugly(
                                max_images=MAX_IMAGES,
                                default_num_images=DEFAULT_NUM_IMAGES,
                                theme=theme)
        with gr.TabItem('유미의 세포들'):
            create_demo_love(
                                max_images=MAX_IMAGES,
                                default_num_images=DEFAULT_NUM_IMAGES,
                                theme=theme)
        with gr.TabItem('기타'):
            create_demo_etc(
                                max_images=MAX_IMAGES,
                                default_num_images=DEFAULT_NUM_IMAGES,
                                theme=theme)
            
block.launch(server_name='0.0.0.0', share=True)