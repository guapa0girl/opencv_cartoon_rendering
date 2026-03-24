import cv2 as cv
import numpy as np

def tomie_style(img):
    # 1. Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Contrast Enhancement
    gray = cv.equalizeHist(gray)
    gray = cv.convertScaleAbs(gray, alpha=1.5, beta=0)

    # 3. Edge Detection (sharp lines)
    edges = cv.Canny(gray, 100, 200)

    # 4. Shading (공포 느낌 핵심)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    shading = cv.divide(gray, blur, scale=255)

    # 5. Texture (grain)
    noise = np.random.normal(0, 10, gray.shape).astype(np.uint8)
    texture = cv.add(shading, noise)

    # 6. Edge 강조
    result = cv.bitwise_and(texture, texture, mask=edges)

    # 7. 최종 대비 강화
    result = cv.convertScaleAbs(result, alpha=1.2, beta=0)

    return result