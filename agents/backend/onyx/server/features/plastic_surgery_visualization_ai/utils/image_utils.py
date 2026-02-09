"""Advanced image utilities using OpenCV and other libraries."""

from typing import Optional, Tuple
from PIL import Image
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

from utils.logger import get_logger

logger = get_logger(__name__)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image (numpy array)
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV not available. Install opencv-python-headless")
    
    # Convert PIL to numpy array
    numpy_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return numpy_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL Image.
    
    Args:
        cv2_image: OpenCV image (numpy array)
        
    Returns:
        PIL Image object
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV not available. Install opencv-python-headless")
    
    # Convert BGR to RGB
    if len(cv2_image.shape) == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(cv2_image)


def resize_image(
    image: Image.Image,
    max_size: Optional[Tuple[int, int]] = None,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image with optional aspect ratio maintenance.
    
    Args:
        image: PIL Image to resize
        max_size: Maximum (width, height). If None, returns original
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if max_size is None:
        return image
    
    if maintain_aspect:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    else:
        image = image.resize(max_size, Image.Resampling.LANCZOS)
    
    return image


def optimize_image_quality(
    image: Image.Image,
    quality: int = 85,
    format: str = "JPEG"
) -> Image.Image:
    """
    Optimize image quality and size.
    
    Args:
        image: PIL Image to optimize
        quality: JPEG quality (1-100)
        format: Output format
        
    Returns:
        Optimized PIL Image
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def detect_faces_opencv(image: np.ndarray) -> list:
    """
    Detect faces in image using OpenCV.
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        List of face bounding boxes [(x, y, w, h), ...]
    """
    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available for face detection")
        return []
    
    try:
        # Load face cascade (requires opencv data files)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces.tolist() if len(faces) > 0 else []
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []


def enhance_image(image: Image.Image, enhancement_type: str = "auto") -> Image.Image:
    """
    Enhance image quality.
    
    Args:
        image: PIL Image to enhance
        enhancement_type: Type of enhancement (auto, contrast, sharpness, color)
        
    Returns:
        Enhanced PIL Image
    """
    from PIL import ImageEnhance
    
    if enhancement_type == "auto":
        # Apply multiple enhancements
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
    elif enhancement_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
    elif enhancement_type == "sharpness":
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
    elif enhancement_type == "color":
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
    
    return image

