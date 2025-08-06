import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO
import torch
import openai_clip as clip

def calculate_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].mean()
    saturation = hsv[:, :, 1].mean()
    brightness = hsv[:, :, 2].mean()
    return round(hue, 2), round(saturation, 2), round(brightness, 2)


def calculate_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return round(contrast, 2)



def get_dominant_colors(img, k=3):
    # img = np.resize(img, (100, 100))
    img = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    return kmeans.cluster_centers_.astype(int)

def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return round(variance, 2)

def detect_scene(img):
    model = YOLO('yolov8n.pt')
    results = model(img)[0]
    labels = results.names
    objs = [labels[int(c)] for c in results.boxes.cls]

    if any(o in objs for o in ['person', 'face']):
        return 'Portrait'
    if any(o in objs for o in ['car', 'bus', 'truck']):
        return 'Urban'
    if any(o in objs for o in ['tree', 'sky', 'plant']):
        return 'Landscape'
    return 'General'


def detect_lightning(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    low = hist[:64].sum()
    high = hist[192:].sum()
    total = hist.sum()
    lightness = np.mean(gray)

    if lightness < 70 or low / total < 0.5:
        return 'Underexposed'
    elif lightness > 200 or high / total > 0.5:
        return 'Overexposed'
    return 'Well-lit'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device = device)

def detect_style(img):
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    styles = ["cinematic photo", "vintage photo", "black and white", 
              "high dynamic range", "portrait", "landscape"]
    text = clip.tokenize(styles).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(input_tensor, text)
    idx = logits_per_image[0].argmax().item()
    return styles[idx]



def color_pallete(img, k=5):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((100, 100))
    arr = np.array(img_pil).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(arr)
    colors = kmeans.cluster_centers_.astype(int)
    hues = np.mean(cv2.cvtColor(colors.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 0])

    if hues < 60:
        harmony = "Warm"
    elif hues > 150:
        harmony = 'Cool'
    else:
        harmony = 'Neutral'
    return colors.tolist(), harmony



def texture_analysis(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if var < 50:
        return 'Soft / Low detail'
    elif var > 300:
        return 'High detail / Sharp'
    return 'Moderate texture'



def detect_faces_and_skin_tone(img):
    if img is None:
        raise ValueError("Image is empty")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    tones = []
    for (x, y, w, h) in faces:
        face_rgb = rgb_img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
        median_hue = np.median(hsv[:, :, 0])
        tones.append(median_hue)

    if not tones:
        return None

    avg_hue = np.mean(tones)
    if avg_hue < 20:
        return 'Warm skin'
    elif avg_hue > 100:
        return 'Cool skin'
    return 'Neutral skin'

    
def metadata(img: np.ndarray):
    hue, sat, bright = calculate_brightness(img)
    contrast = calculate_contrast(img)
    dominant_colors = get_dominant_colors(img, k=3)
    sharpness = calculate_sharpness(img)
    scene = detect_scene(img)
    lighting = detect_lightning(img)
    style = detect_style(img)
    palette = color_pallete(img)
    texture = texture_analysis(img)
    s_tones = detect_faces_and_skin_tone(img)

    metainfo = f"Hue: {hue}, Saturation: {sat}, Brightness: {bright}, Contrast: {contrast}, Sharpness: {sharpness}, Dominant Colors: {dominant_colors}, Scene: {scene}, Lighting: {lighting}, Style: {style}, Color Palette: {palette}, Texture: {texture}, Skin Tones: {s_tones}"
    return metainfo



if __name__ == "__main__":
    path = "assets/face.png"
    # print("Brightness:", calculate_brightness(path))
    # print("Contrast:", calculate_contrast(path))
    print("Metadata: ", metadata(path))

