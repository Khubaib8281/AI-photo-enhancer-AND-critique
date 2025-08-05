from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

def apply_teal_orange(img):
    arr = np.array(img).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 1.1, 0, 255)  # Blue boost
    arr[..., 1] = np.clip(arr[..., 1] * 0.95, 0, 255) # Green tone
    arr[..., 2] = np.clip(arr[..., 2] * 1.2, 0, 255)  # Red boost
    return Image.fromarray(arr.astype(np.uint8))


def apply_sepia(img):
    arr = np.array(img)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    sepia = np.zeros_like(arr)
    sepia[..., 0] = np.clip(0.393 * r + 0.769 * g + 0.189 * b, 0, 255)
    sepia[..., 1] = np.clip(0.349 * r + 0.686 * g + 0.168 * b, 0, 255)
    sepia[..., 2] = np.clip(0.272 * r + 0.534 * g + 0.131 * b, 0, 255)
    return Image.fromarray(sepia.astype(np.uint8))



from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

def add_vignette(img: Image.Image, strength: float = 1.5, blur_radius: int = 200) -> Image.Image:
    """
    Adds a professional-style vignette effect (dark corners, bright center).
    
    Args:
        img: Input image (PIL Image).
        strength: Multiplier to control the darkness of corners (1.0 = mild, >1.5 = strong).
        blur_radius: How smooth the vignette transition is.

    Returns:
        Image with vignette applied.
    """
    width, height = img.size
    center_x, center_y = width / 2, height / 2

    # Create radial gradient using distance from center
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = dist / max_dist  # Normalize to 0–1

    # Calculate vignette mask: brighter at center, darker at edges
    mask = 1 - dist  # Center is 1, edges 0
    mask = np.clip(mask * strength, 0, 1)  # Amplify and clip
    mask = (mask * 255).astype(np.uint8)

    # Convert mask to image and blur
    vignette_mask = Image.fromarray(mask, mode='L').filter(ImageFilter.GaussianBlur(blur_radius))

    # Convert to 3-channel alpha mask
    vignette_mask_rgb = vignette_mask.convert("RGB")

    # Composite with black background
    black = Image.new("RGB", img.size, "black")
    result = Image.composite(img, black, vignette_mask)

    return result

def apply_edits(image: Image.Image, actions: dict) -> Image.Image:
    img = image.convert("RGB")


    # 1. Contrast
    if actions.get("increase_contrast"):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

    if actions.get("reduce_contrast") == "slightly":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.9)

    # 2. Brightness
    if actions.get("decrease_brightness") == "slightly":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)

    if actions.get("boost_brightness") == "moderate":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)

    # 3. Saturation
    if actions.get("increase_saturation") == "high":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(2.0)

    if actions.get("desaturate") == "moderate":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.5)

    # 4. Vignette (simple radial darken)
    if actions.get("add_vignette"):
        img = add_vignette(img)

    # 5. Color Grade
    if actions.get("color_grade") == "teal_orange":
        img = apply_teal_orange(img)

    # 6. Filters
    if actions.get("add_filter") == "sepia":
        img = apply_sepia(img)

    # 7. White Balance (placeholder: auto white fix)
    if actions.get("balance_white"):
        img = ImageOps.autocontrast(img)

    # 8. Sharpen
    if actions.get("sharpen"):
        img = img.filter(ImageFilter.SHARPEN)

    return img