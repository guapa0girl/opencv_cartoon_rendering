import cv2 as cv
import numpy as np

def tomie_style(img):
    # --- 1. 전처리 (Grayscale 및 기초 면 정리) ---
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Bilateral Filter로 피부는 매끈하게, 에지(눈매, 머리카락)는 날카롭게 유지
    # 이 필터는 피부의 잡티를 제거하면서 이목구비의 섬세한 경계는 보존합니다.
    gray_clean = cv.bilateralFilter(gray, 7, 50, 50)

    # --- 2. 펜선 추출 (두꺼우면서도 섬세한 라인) ---
    # Adaptive Threshold를 사용하여, 미세한 머리카락 결, 눈썹, 입술 윤곽을 잡아냅니다.
    # 블록 사이즈를 작게(7) 주어 이토 준지 특유의 얇고 신경질적인 선을 강조합니다.
    fine_edges = cv.adaptiveThreshold(gray_clean, 255,
                                       cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, 7, 2)
    
    # 머리카락 뭉치감을 만화처럼 표현하기 위해 선을 두껍게 강조 (Dilation)
    # 255 - fine_edges는 검은색 바탕에 흰색 선으로 만듭니다.
    # kernel 사이즈와 iteration을 조절하여 선의 굵기를 정합니다.
    # `image_0.png`처럼 두꺼운 머리카락 뭉치감을 위해 (3, 3) 커널을 사용합니다.
    kernel_dil = np.ones((3, 3), np.uint8)
    dilated_edges = cv.dilate(255 - fine_edges, kernel_dil, iterations=1)
    # 다시 반전시켜 흰색 바탕에 검은색 굵은 선으로 만듭니다.
    thick_edges = 255 - dilated_edges

    # --- 3. 강렬한 흑백 대비 (토미에의 고귀함과 기괴함) ---
    # `image_0.png`처럼 피부는 완전히 하얗게 날리고 머리카락만 어둡게 표현하기 위해 
    # Global Threshold를 적용합니다. 130 이하의 어두운 영역만 검게 남습니다.
    # 이 단계에서 중간 톤(회색)은 모두 제거되어 흑과 백의 명확한 경계가 생깁니다.
    _, high_contrast = cv.threshold(gray_clean, 130, 255, cv.THRESH_BINARY)

    # --- 4. 선과 면의 최종 합성 (만화 드로잉) ---
    # 날카로운 선 패턴(`fine_edges`), 두꺼운 선 패턴(`thick_edges`), 
    # 그리고 강한 흑백 면 패턴(`high_contrast`)을 합성합니다.
    # 결과는 흰 바탕에 검은 선과 명암만 남은 형태가 되어, 원작 만화와 유사한 스타일이 됩니다.
    sketch = cv.bitwise_and(thick_edges, high_contrast)
    result = cv.bitwise_and(sketch, fine_edges)

    # --- 5. 최종 디테일 강조 ---
    # 눈동자와 입술 주변을 더 선명하게 만들기 위해 디테일을 한번 더 강조합니다.
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv.filter2D(result, -1, kernel_sharpen)
    
    # 선명하게 만든 이미지와 최종 이진화 이미지를 합성
    result = cv.bitwise_and(sharp, result)

    # 마지막 대비 강조
    result = cv.convertScaleAbs(result, alpha=1.2, beta=0)

    return result