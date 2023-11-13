# Webtoon_InST

## Getting Started

We recommend running our code using:

- NVIDIA GPU + CUDA, CuDNN
- Python 3, Anaconda

<p align="right">(<a href="#top">back to top</a>)</p>

### 1. Installation

Clone the repositories.
   ```sh
   git clone https://github.com/ssojeong/Webtoon_InST.git
   git clone https://github.com/zyxElsa/InST.git
   ```

Run following commands to install necessary packages.
  ```sh
  conda env create -f environment.yaml
  conda activate ldm
  ```
<p align="right">(<a href="#top">back to top</a>)</p>

### 2. Pretrained Models for Webtoon_InST Inference
Download the pretrained models and save it to the indicated location.

| Pretrained Model | Save Location | Reference Repo/Source.
|---|---|---
| [Stable Diffusion](https://github.com/CompVis/stable-diffusion.git) | ./InST/models/sd/sd-v1-4.ckpt. | [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion.git)
| [YeosinGangrim](https://drive.google.com/drive/folders/1x0XIFSX6cKO3bjdaI3JOdUtLppqf9Qmy?usp=sharing) | ./InST/logs/yeosin/ | [여신강림-네이버웹툰](https://comic.naver.com/webtoon/list?titleId=703846)
| [UglyPeoples](https://drive.google.com/drive/folders/1IQzcxdi8F2nAQaiZwtPyEqimt_UaZtH9?usp=sharing) | ./InST/logs/ugly/ | [어글리피플즈-네이버웹툰](https://comic.naver.com/webtoon/list?titleId=732953)
| [YumiSepo](https://drive.google.com/drive/folders/1CI4e3Px_AC1ZIJokTtkF1wrjq2jYkVp4?usp=sharing) | ./InST/logs/yumi/ | [유미의세포-네이버웹툰](https://series.naver.com/comic/detail.series?productNo=3900477)
| [Other style](https://drive.google.com/drive/folders/141l8dvD_tR7z2uqqnPwiPUht4Gukcge0?usp=sharing) | ./InST/logs/etc/ | [An Image in the InST(CVPR, 2023) paper](https://arxiv.org/abs/2211.13203)
<p align="right">(<a href="#top">back to top</a>)</p>



### 3. Implementation
Run following commands and open the shared link.
  ```sh
  python demo_canny.py
  ```
- The Gradio app allows you to change hyperparameters(steps, style guindace sclae, etc.)
- The [FFHQ](https://github.com/NVlabs/ffhq-dataset.git) sample datasets has been uploaded in the `./data/face`, so you can use it for testing.
<p align="right">(<a href="#top">back to top</a>)</p>


### cf. Different style guidance scales for background and foreground
If you want to give different style guidance to the background and foreground, clone the repository below and use it.
  ```sh
  git clone https://github.com/xuebinqin/DIS.git
  ```
The Implementation code is already in this inference python file, but the detailed implementation method will be updated later.
<p align="right">(<a href="#top">back to top</a>)</p>
