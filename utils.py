import cv2 as cv
import numpy as np

def tomie_style(img):
    # 1. Grayscale 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. 노이즈 제거 (피부 톤을 깨끗하게 정리)
    gray_blurred = cv.bilateralFilter(gray, 9, 75, 75)

    # 3. 선 추출 (Canny 대신 Sketch 효과를 위해 Adaptive Threshold 사용)
    # 토미에 특유의 얇고 날카로운 선을 따기 위해 블록 사이즈를 작게 조절합니다.
    edges = cv.adaptiveThreshold(gray_blurred, 255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 9, 2)

    # 4. 강한 대비 (Thresholding) - 배경과 피부를 하얗게 날림
    # 127보다 밝은 부분은 255(흰색)로, 어두운 부분은 강조
    _, high_contrast = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)

    # 5. 선과 면의 합성
    # 에지(edges)와 고대비 이미지(high_contrast)를 합성하여 
    # 흰 바탕에 검은 선과 명암만 남깁니다.
    result = cv.bitwise_and(edges, high_contrast)

    # 6. (선택) 질감 추가 - 이토 준지 특유의 스크린톤 느낌
    noise = np.random.randint(0, 20, result.shape, dtype='uint8')
    # 어두운 영역에만 아주 살짝 노이즈를 섞어 종이 질감을 줍니다.
    result = cv.addWeighted(result, 0.95, noise, 0.05, 0)

    return result