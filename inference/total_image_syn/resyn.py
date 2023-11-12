from PIL import Image, ImageDraw, ImageFont
import os

def print_final(save_path, outdir):
    # 하얀 바탕 이미지 생성
    width, height = 500, 760  # 하얀 바탕 이미지의 가로 및 세로 크기
    white_image = Image.new("RGB", (width, height), "white")

    # 특정 디렉토리에서 이미지 불러오기
    image = Image.open(save_path)
    w, h = image.size
    w, h = float(w), float(h)
    w, h = int(w / 0.8), int(h / 0.8)
    # 이미지 크기 조정 (하얀 바탕 이미지 크기에 맞게)
    image = image.resize((w, h))  # 이미지 크기 조정 (절반 크기)
    w_m = int((width - w)/2)

    #logo load
    logo_path = './inference/total_image_syn/logo.png'
    logo = Image.open(logo_path)
    w_l, h_l = logo.size
    w_l, h_l = float(w_l), float(h_l)
    w_l, h_l = int(w_l / 4), int(h_l / 4)
    logo = logo.resize((w_l, h_l))


    # 하얀 바탕 이미지에 이미지 붙여넣기
    x, y = w_m, 60  # 붙여넣을 위치 (x, y 좌표)
    x_l, y_l = width - 190, 640

    white_image.paste(image, (x, y))
    white_image.paste(logo, (x_l, y_l))


    # 텍스트 추가
    draw = ImageDraw.Draw(white_image)
    text = "http://ml.cau.ac.kr"  # 삽입할 텍스트
    font_size = 17
    font_path = "./inference/total_image_syn/NANUMSQUARE.TTF"  # 실제 폰트 파일의 경로로 변경
    font = ImageFont.truetype(font_path, font_size)  # 폰트 경로 사용
    #font = ImageFont.load_default()  # 기본 폰트 사용 (다른 폰트 사용 가능)

    # 텍스트 위치 계산
    text_x = (width - 10) // 2 + 72  # 가운데 정렬
    text_y = (height - 10) - 60  # 이미지 아래에 위치

    # 텍스트 삽입
    draw.text((text_x, text_y), text, fill="black", font=font)

    # 결과 이미지 저장
    path = os.path.join(outdir, 'print_result.jpg')
    white_image.save(path)
