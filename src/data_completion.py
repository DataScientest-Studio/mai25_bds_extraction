from PIL import Image, UnidentifiedImageError
import os
import cv2
from tqdm import tqdm

def get_images_to_fix(paths):
    """
    Check a list of image file paths and return those that are missing, corrupted or unreadable.

    Parameters:
        paths (Iterable[str]): A list or iterable of image file paths.

    Returns:
        List[str]: A list of file paths corresponding to missing, corrupted or unreadable images.
    """
    result = []
    for path in tqdm(paths):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            result.append(path)
    return result

def convert_iit_to_rvl_tiff(image_path: str, output_path: str) -> str:
    """
    Convert an IIT-CDIP image to RVL-CDIP format:
    - Grayscale
    - Resized to have max dimension = 1000 px (aspect ratio preserved)
    - Saved as TIFF with LZW compression (safe for 8-bit grayscale)

    Parameters:
    -----------
    image_path : str
        Path to input image (e.g., IIT-CDIP .tif file)

    output_path : str
        Path to save the converted image (must end with .tif or .tiff)

    Returns:
    --------
    str : path to the saved TIFF file
    """
    try:
        img = Image.open(image_path).convert("L")
    except (UnidentifiedImageError, OSError):
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            raise ValueError(f"Cannot open image with PIL or OpenCV: {image_path}")
        img = Image.fromarray(img_cv)

    # Resize with aspect ratio preserved
    w, h = img.size
    if w >= h:
        new_w = 1000
        new_h = int((1000 / w) * h)
    else:
        new_h = 1000
        new_w = int((1000 / h) * w)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Force TIFF format with safe compression
    img_resized.save(output_path, format="TIFF", compression="tiff_lzw")

    return output_path
