import cv2 as cv
import os
from utils import tomie_style

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_images():
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"tomie_{filename}")

            print(f"[DEBUG] loading: {input_path}")

            img = cv.imread(input_path)

            if img is None:
                print(f"❌ Failed to load {filename}")
                continue

            result = tomie_style(img)

            cv.imwrite(output_path, result)
            print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    process_images()