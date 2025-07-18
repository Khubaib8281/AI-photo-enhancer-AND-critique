import cv2
import numpy as np
from PIL import Image

def calculate_brightness(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0].mean()
    saturation = hsv[:, :, 1].mean()
    brightness = hsv[:, :, 2].mean()
    return round(hue, 2), round(saturation, 2), round(brightness, 2)

def calculate_contrast(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast = image.std()
    return round(contrast, 2)


if __name__ == "__main__":
    path = "assets/sample.jpg"
    print("Brightness:", calculate_brightness(path))
    print("Contrast:", calculate_contrast(path))

