import sys
sys.path.append("./InST")
sys.path.append('./DIS/IS_Net')
sys.path.append('./inference/total_image_syn')

from diffusers.utils import load_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from diffusers.utils import load_image
import cv2
import argparse

#배경 분리
from DIS.IS_Net.Inference import *
from DIS.IS_Net.models.isnet import ISNetGTEncoder, ISNetDIS

#inference
from inference_guided import *

#로고 이미지
from resyn import *

def resize_prev(img, basewidth):
    basewidth = basewidth
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.LANCZOS)
    return img


def centor_crop(img, size=512):
    width, height = img.size   # Get dimensions
    new_width, new_height = size, size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img


def image_to_image_mask(outdir, emb, style_file, content_dir, fg_wt, bg_wt, name):
    
    sys.path.append(os.getcwd())
    os.makedirs(outdir, exist_ok=True)

    #1. mask 생성
    mask_dir = os.path.join(outdir, 'mask')
    content_dir = content_dir
    _, mask_file = mask_main(content_dir, mask_dir)
    #output: mask 경로

    #2. cv2: fore_file, back_file 생성
    if os.path.isdir(content_dir):
        list = os.listdir(content_dir)[0]
        image = load_image(os.path.join(content_dir, list))
    else:
        image = load_image(content_dir)
    #이미지 사이즈 조절
    image = resize_prev(image, 512)
    image = centor_crop(image)
    src = np.array(image)

    #mask 사이즈 조절
    mask = load_image(mask_file)
    mask = resize_prev(mask, 512)
    mask = centor_crop(mask)
    mask = np.array(mask)
    
    horse_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    _, horse_mask = cv2.threshold(horse_mask,127,255,cv2.THRESH_BINARY)
    horse_mask = np.repeat(horse_mask[:,:,np.newaxis],3,-1)

    fore = cv2.bitwise_and(src, horse_mask)
    print('cv2 후 fore size:', np.array(fore).shape)
    back = cv2.bitwise_and(src, 255-horse_mask)
    print('cv2 후 back size:', np.array(back).shape)

    #색깔 변화 후 저장
    fore_dir, back_dir = os.path.join(outdir, 'fore'), os.path.join(outdir, 'back')
    os.makedirs(fore_dir, exist_ok=True)
    os.makedirs(back_dir, exist_ok=True)

    name = mask_file.split('/')[-1].split('.')[0]+'_'+name
    fore_file, back_file = os.path.join(fore_dir, name+'.png'), os.path.join(back_dir, name+'.png')

    cv2.cvtColor(fore, cv2.COLOR_BGR2RGB, dst=fore)
    cv2.imwrite(fore_file, fore)

    cv2.cvtColor(back, cv2.COLOR_BGR2RGB, dst=back)
    cv2.imwrite(back_file, back)


    #3. style 변형
    content_dir, style_file, weight, emb = fore_file, style_file, fg_wt, emb
    _, fore_file, count = style_start_org(content_dir, style_file, weight, emb, outdir=outdir) 

    content_dir, style_file, weight, emb = back_file, style_file, bg_wt, emb
    _, back_file, count = style_start_org(content_dir, style_file, weight, emb, outdir=outdir)

    #각각 생성된 이미지 합치기
    image = load_image(fore_file)
    fore = np.array(image)

    back = load_image(back_file)
    back = np.array(back)

    #copyTo(원본, 마스크, 옮길 사진)
    print('fore size: ', np.array(fore).shape)
    print('horse_mask:', np.array(horse_mask).shape)
    print('back:', np.array(back).shape)

    #4. 최종 이미지
    dst3 = cv2.copyTo(fore, horse_mask, back)
    dst3 = np.array(dst3)

    #저장
    cv2.cvtColor(dst3, cv2.COLOR_BGR2RGB, dst=dst3)
    outdir = os.path.join(outdir, 'merge')
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, f'{name}_fg:{fg_wt}_bg:{bg_wt}_merge.png'),dst3)
    cv2.cvtColor(dst3, cv2.COLOR_BGR2RGB, dst=dst3)
    
    return dst3


def image_to_image_canny(outdir, emb, style_file, content_dir, fg_wt, name, ddim_steps, custom=50):
    
    sys.path.append(os.getcwd())
    os.makedirs(outdir, exist_ok=True)

    #1. load 이미지 
    if os.path.isdir(content_dir):
        list = os.listdir(content_dir)[0]
        image = load_image(os.path.join(content_dir, list))
    else:
        image = load_image(content_dir)
    
    #이미지 사이즈 조절
    image = resize_prev(image, 512)
    image = centor_crop(image)
    src = np.array(image)
    
    org_outpath =  os.path.join(outdir, 'original')
    os.makedirs(org_outpath, exist_ok=True)
    base_count = len(os.listdir(org_outpath))
    org_file = os.path.join(org_outpath, name+'-'+f'{base_count:04}'+'.png')
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB, dst=src)
    cv2.imwrite(org_file, src)
    
    #2. style 변형(with canny - style_start_guide)
    _, total_file, count = style_start_guide(org_file, style_file, fg_wt, emb, outdir=outdir, ddim_steps=ddim_steps, custom=custom) #args 넣기
    image = load_image(total_file)
    dst3 = np.array(image)

    
    #3. 최종 출력물 실행
    print_final(total_file, outdir)
    
    return dst3
    
    # #cf. 이미지 사이즈 조절: 검은 테두리
    # image = dst3

    # width, height = 512, 512
    # transparent_image = np.zeros((height, width, 4), dtype=np.uint8)  # 4 채널 이미지 (투명도 포함)

    # # 이미지 크기 조정 (투명 이미지 크기에 맞게)
    # new_width, new_height = 400, 400  # 원하는 작은 크기
    # image = cv2.resize(image, (new_width, new_height))

    # # 이미지를 투명 이미지 위에 삽입
    # x, y = (width - new_width) // 2, (height - new_height) // 2
    # transparent_image[y:y+new_height, x:x+new_width, :3] = image  # 마지막 채널 (투명도) 제외

    # return transparent_image[:,:,:3]


def image_to_image_org(outdir, emb, style_file, content_dir, fg_wt, name, ddim_steps):
    
    sys.path.append(os.getcwd())
    os.makedirs(outdir, exist_ok=True)

    #1. load 이미지 
    if os.path.isdir(content_dir):
        list = os.listdir(content_dir)[0]
        image = load_image(os.path.join(content_dir, list))
    else:
        image = load_image(content_dir)
    #이미지 사이즈 조절
    image = resize_prev(image, 512)
    image = centor_crop(image)
    src = np.array(image)
    
    org_file = os.path.join(outdir, name+'.png')
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB, dst=src)
    cv2.imwrite(org_file, src)

    #2. style 변형(with canny - style_start_org)
    _, total_file, count = style_start_org(org_file, style_file, fg_wt, emb, outdir=outdir, ddim_steps=ddim_steps) #args 넣기
    image = load_image(
        total_file)
    dst3 = np.array(image)
    
    #저장
    cv2.cvtColor(dst3, cv2.COLOR_BGR2RGB, dst=dst3)
    outdir = os.path.join(outdir, 'merge')
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, count+'-'+f'{name}_fg:{fg_wt}_merge.png'),dst3)
    cv2.cvtColor(dst3, cv2.COLOR_BGR2RGB, dst=dst3)
    
    return dst3