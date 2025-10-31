"""
Gamma App - Image Utilities
Advanced image processing and manipulation utilities
"""

import io
import base64
from typing import Tuple, Optional, List, Dict, Any, Union
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced image processing class"""
    
    def __init__(self):
        self.supported_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF', 'WEBP']
        self.max_size = (4096, 4096)
        self.quality = 85
    
    def resize_image(
        self,
        image: Image.Image,
        size: Tuple[int, int],
        maintain_aspect: bool = True,
        resample: int = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """Resize image with optional aspect ratio maintenance"""
        try:
            if maintain_aspect:
                image.thumbnail(size, resample)
                return image
            else:
                return image.resize(size, resample)
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
    
    def crop_image(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Crop image to specified box"""
        try:
            return image.crop(box)
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            raise
    
    def rotate_image(
        self,
        image: Image.Image,
        angle: float,
        expand: bool = True
    ) -> Image.Image:
        """Rotate image by specified angle"""
        try:
            return image.rotate(angle, expand=expand)
        except Exception as e:
            logger.error(f"Error rotating image: {e}")
            raise
    
    def flip_image(
        self,
        image: Image.Image,
        direction: str = 'horizontal'
    ) -> Image.Image:
        """Flip image horizontally or vertically"""
        try:
            if direction == 'horizontal':
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif direction == 'vertical':
                return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                raise ValueError("Direction must be 'horizontal' or 'vertical'")
        except Exception as e:
            logger.error(f"Error flipping image: {e}")
            raise
    
    def adjust_brightness(
        self,
        image: Image.Image,
        factor: float
    ) -> Image.Image:
        """Adjust image brightness"""
        try:
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.error(f"Error adjusting brightness: {e}")
            raise
    
    def adjust_contrast(
        self,
        image: Image.Image,
        factor: float
    ) -> Image.Image:
        """Adjust image contrast"""
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.error(f"Error adjusting contrast: {e}")
            raise
    
    def adjust_saturation(
        self,
        image: Image.Image,
        factor: float
    ) -> Image.Image:
        """Adjust image saturation"""
        try:
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.error(f"Error adjusting saturation: {e}")
            raise
    
    def apply_filter(
        self,
        image: Image.Image,
        filter_type: str
    ) -> Image.Image:
        """Apply image filter"""
        try:
            filters = {
                'blur': ImageFilter.BLUR,
                'contour': ImageFilter.CONTOUR,
                'detail': ImageFilter.DETAIL,
                'edge_enhance': ImageFilter.EDGE_ENHANCE,
                'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
                'emboss': ImageFilter.EMBOSS,
                'find_edges': ImageFilter.FIND_EDGES,
                'smooth': ImageFilter.SMOOTH,
                'smooth_more': ImageFilter.SMOOTH_MORE,
                'sharpen': ImageFilter.SHARPEN,
            }
            
            if filter_type not in filters:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            return image.filter(filters[filter_type])
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            raise
    
    def add_watermark(
        self,
        image: Image.Image,
        watermark_text: str,
        position: str = 'bottom_right',
        opacity: float = 0.5,
        font_size: int = 20
    ) -> Image.Image:
        """Add text watermark to image"""
        try:
            # Create a copy of the image
            watermarked = image.copy()
            
            # Create a transparent overlay
            overlay = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            if position == 'top_left':
                x, y = 10, 10
            elif position == 'top_right':
                x, y = watermarked.width - text_width - 10, 10
            elif position == 'bottom_left':
                x, y = 10, watermarked.height - text_height - 10
            elif position == 'bottom_right':
                x, y = watermarked.width - text_width - 10, watermarked.height - text_height - 10
            elif position == 'center':
                x, y = (watermarked.width - text_width) // 2, (watermarked.height - text_height) // 2
            else:
                x, y = 10, 10
            
            # Draw text with opacity
            alpha = int(255 * opacity)
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, alpha))
            
            # Composite the overlay onto the image
            if watermarked.mode != 'RGBA':
                watermarked = watermarked.convert('RGBA')
            
            watermarked = Image.alpha_composite(watermarked, overlay)
            
            return watermarked.convert('RGB')
        except Exception as e:
            logger.error(f"Error adding watermark: {e}")
            raise
    
    def add_image_watermark(
        self,
        image: Image.Image,
        watermark_image: Image.Image,
        position: str = 'bottom_right',
        opacity: float = 0.5,
        scale: float = 0.1
    ) -> Image.Image:
        """Add image watermark to image"""
        try:
            # Resize watermark
            watermark_size = (
                int(image.width * scale),
                int(image.height * scale)
            )
            watermark_image = watermark_image.resize(watermark_size, Image.Resampling.LANCZOS)
            
            # Convert to RGBA if needed
            if watermark_image.mode != 'RGBA':
                watermark_image = watermark_image.convert('RGBA')
            
            # Adjust opacity
            if opacity < 1.0:
                alpha = watermark_image.split()[-1]
                alpha = alpha.point(lambda p: int(p * opacity))
                watermark_image.putalpha(alpha)
            
            # Calculate position
            if position == 'top_left':
                x, y = 10, 10
            elif position == 'top_right':
                x, y = image.width - watermark_size[0] - 10, 10
            elif position == 'bottom_left':
                x, y = 10, image.height - watermark_size[1] - 10
            elif position == 'bottom_right':
                x, y = image.width - watermark_size[0] - 10, image.height - watermark_size[1] - 10
            elif position == 'center':
                x, y = (image.width - watermark_size[0]) // 2, (image.height - watermark_size[1]) // 2
            else:
                x, y = 10, 10
            
            # Create a copy of the image
            watermarked = image.copy()
            if watermarked.mode != 'RGBA':
                watermarked = watermarked.convert('RGBA')
            
            # Paste watermark
            watermarked.paste(watermark_image, (x, y), watermark_image)
            
            return watermarked.convert('RGB')
        except Exception as e:
            logger.error(f"Error adding image watermark: {e}")
            raise
    
    def create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (300, 300),
        quality: int = 85
    ) -> Image.Image:
        """Create thumbnail of image"""
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return thumbnail
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            raise
    
    def convert_format(
        self,
        image: Image.Image,
        format: str,
        quality: int = 85
    ) -> Image.Image:
        """Convert image to different format"""
        try:
            if format.upper() not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            # Convert to RGB if saving as JPEG
            if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Error converting format: {e}")
            raise
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get image information"""
        try:
            return {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'width': image.width,
                'height': image.height,
                'aspect_ratio': image.width / image.height,
                'has_transparency': image.mode in ('RGBA', 'LA', 'P')
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}
    
    def detect_faces(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using OpenCV"""
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Convert back to PIL format
            face_boxes = []
            for (x, y, w, h) in faces:
                face_boxes.append((x, y, x + w, y + h))
            
            return face_boxes
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def blur_faces(
        self,
        image: Image.Image,
        blur_radius: int = 20
    ) -> Image.Image:
        """Blur detected faces in image"""
        try:
            faces = self.detect_faces(image)
            if not faces:
                return image
            
            blurred_image = image.copy()
            
            for face_box in faces:
                # Crop face region
                face_region = image.crop(face_box)
                
                # Blur face region
                blurred_face = face_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                # Paste blurred face back
                blurred_image.paste(blurred_face, face_box)
            
            return blurred_image
        except Exception as e:
            logger.error(f"Error blurring faces: {e}")
            raise
    
    def create_collage(
        self,
        images: List[Image.Image],
        layout: str = 'grid',
        spacing: int = 10,
        background_color: str = 'white'
    ) -> Image.Image:
        """Create image collage"""
        try:
            if not images:
                raise ValueError("No images provided")
            
            if layout == 'grid':
                return self._create_grid_collage(images, spacing, background_color)
            elif layout == 'horizontal':
                return self._create_horizontal_collage(images, spacing, background_color)
            elif layout == 'vertical':
                return self._create_vertical_collage(images, spacing, background_color)
            else:
                raise ValueError("Invalid layout type")
        except Exception as e:
            logger.error(f"Error creating collage: {e}")
            raise
    
    def _create_grid_collage(
        self,
        images: List[Image.Image],
        spacing: int,
        background_color: str
    ) -> Image.Image:
        """Create grid collage"""
        # Calculate grid dimensions
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Resize all images to same size
        target_size = (200, 200)
        resized_images = []
        for img in images:
            resized = img.copy()
            resized.thumbnail(target_size, Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # Calculate collage size
        collage_width = cols * target_size[0] + (cols - 1) * spacing
        collage_height = rows * target_size[1] + (rows - 1) * spacing
        
        # Create collage
        collage = Image.new('RGB', (collage_width, collage_height), background_color)
        
        # Paste images
        for i, img in enumerate(resized_images):
            row = i // cols
            col = i % cols
            x = col * (target_size[0] + spacing)
            y = row * (target_size[1] + spacing)
            collage.paste(img, (x, y))
        
        return collage
    
    def _create_horizontal_collage(
        self,
        images: List[Image.Image],
        spacing: int,
        background_color: str
    ) -> Image.Image:
        """Create horizontal collage"""
        # Resize all images to same height
        target_height = 200
        resized_images = []
        for img in images:
            aspect_ratio = img.width / img.height
            target_width = int(target_height * aspect_ratio)
            resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # Calculate collage size
        collage_width = sum(img.width for img in resized_images) + (len(resized_images) - 1) * spacing
        collage_height = target_height
        
        # Create collage
        collage = Image.new('RGB', (collage_width, collage_height), background_color)
        
        # Paste images
        x = 0
        for img in resized_images:
            collage.paste(img, (x, 0))
            x += img.width + spacing
        
        return collage
    
    def _create_vertical_collage(
        self,
        images: List[Image.Image],
        spacing: int,
        background_color: str
    ) -> Image.Image:
        """Create vertical collage"""
        # Resize all images to same width
        target_width = 200
        resized_images = []
        for img in images:
            aspect_ratio = img.width / img.height
            target_height = int(target_width / aspect_ratio)
            resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # Calculate collage size
        collage_width = target_width
        collage_height = sum(img.height for img in resized_images) + (len(resized_images) - 1) * spacing
        
        # Create collage
        collage = Image.new('RGB', (collage_width, collage_height), background_color)
        
        # Paste images
        y = 0
        for img in resized_images:
            collage.paste(img, (0, y))
            y += img.height + spacing
        
        return collage
    
    def image_to_base64(self, image: Image.Image, format: str = 'JPEG') -> str:
        """Convert image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format, quality=self.quality)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string to image"""
        try:
            image_data = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            raise
    
    def save_image(
        self,
        image: Image.Image,
        file_path: str,
        format: str = 'JPEG',
        quality: int = 85
    ) -> bool:
        """Save image to file"""
        try:
            # Convert format if needed
            if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            image.save(file_path, format=format, quality=quality)
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

# Global image processor instance
image_processor = ImageProcessor()

def resize_image(image: Image.Image, size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """Resize image using global processor"""
    return image_processor.resize_image(image, size, maintain_aspect)

def crop_image(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop image using global processor"""
    return image_processor.crop_image(image, box)

def rotate_image(image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
    """Rotate image using global processor"""
    return image_processor.rotate_image(image, angle, expand)

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness using global processor"""
    return image_processor.adjust_brightness(image, factor)

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image contrast using global processor"""
    return image_processor.adjust_contrast(image, factor)

def apply_filter(image: Image.Image, filter_type: str) -> Image.Image:
    """Apply image filter using global processor"""
    return image_processor.apply_filter(image, filter_type)

def add_watermark(image: Image.Image, watermark_text: str, position: str = 'bottom_right', opacity: float = 0.5) -> Image.Image:
    """Add watermark using global processor"""
    return image_processor.add_watermark(image, watermark_text, position, opacity)

def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Create thumbnail using global processor"""
    return image_processor.create_thumbnail(image, size)

def detect_faces(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """Detect faces using global processor"""
    return image_processor.detect_faces(image)

def blur_faces(image: Image.Image, blur_radius: int = 20) -> Image.Image:
    """Blur faces using global processor"""
    return image_processor.blur_faces(image, blur_radius)

























