"""
Image Comparison Utilities
===========================

Utilities for comparing original and upscaled images.
"""

import logging
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


class ImageComparison:
    """
    Create side-by-side comparisons of original and upscaled images.
    
    Features:
    - Side-by-side layout
    - Before/after labels
    - Metrics overlay
    - Grid layouts
    """
    
    @staticmethod
    def create_side_by_side(
        original: Image.Image,
        upscaled: Image.Image,
        labels: Optional[Tuple[str, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        spacing: int = 20
    ) -> Image.Image:
        """
        Create side-by-side comparison image.
        
        Args:
            original: Original image
            upscaled: Upscaled image
            labels: Optional tuple of (original_label, upscaled_label)
            metrics: Optional metrics to display
            spacing: Spacing between images
            
        Returns:
            Comparison image
        """
        labels = labels or ("Original", "Upscaled")
        
        # Resize original to match upscaled height for comparison
        orig_resized = original.resize(
            (upscaled.width, upscaled.height),
            Image.Resampling.LANCZOS
        )
        
        # Calculate dimensions
        img_width = orig_resized.width
        img_height = orig_resized.height
        total_width = img_width * 2 + spacing * 3
        total_height = img_height + spacing * 2 + 40  # Extra space for labels
        
        # Create canvas
        canvas = Image.new("RGB", (total_width, total_height), color="white")
        
        # Paste images
        x_offset = spacing
        y_offset = 40  # Space for labels
        
        canvas.paste(orig_resized, (x_offset, y_offset))
        canvas.paste(upscaled, (x_offset + img_width + spacing, y_offset))
        
        # Add labels
        draw = ImageDraw.Draw(canvas)
        
        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                font = ImageFont.load_default()
        
        # Draw labels
        label_y = 10
        draw.text(
            (x_offset, label_y),
            labels[0],
            fill="black",
            font=font
        )
        draw.text(
            (x_offset + img_width + spacing, label_y),
            labels[1],
            fill="black",
            font=font
        )
        
        # Add metrics if provided
        if metrics:
            metrics_text = []
            if "quality_score" in metrics:
                metrics_text.append(f"Quality: {metrics['quality_score']:.3f}")
            if "ssim" in metrics:
                metrics_text.append(f"SSIM: {metrics['ssim']:.3f}")
            if "psnr" in metrics:
                metrics_text.append(f"PSNR: {metrics['psnr']:.1f} dB")
            
            if metrics_text:
                metrics_str = " | ".join(metrics_text)
                text_width = draw.textlength(metrics_str, font=font)
                metrics_x = (total_width - text_width) // 2
                draw.text(
                    (metrics_x, total_height - 30),
                    metrics_str,
                    fill="gray",
                    font=font
                )
        
        return canvas
    
    @staticmethod
    def create_grid(
        images: list[Image.Image],
        labels: Optional[list[str]] = None,
        cols: int = 2,
        spacing: int = 10
    ) -> Image.Image:
        """
        Create grid comparison of multiple images.
        
        Args:
            images: List of images
            labels: Optional list of labels
            cols: Number of columns
            spacing: Spacing between images
            
        Returns:
            Grid image
        """
        if not images:
            raise ValueError("No images provided")
        
        labels = labels or [f"Image {i+1}" for i in range(len(images))]
        
        # Resize all images to same size
        target_size = images[0].size
        resized_images = [
            img.resize(target_size, Image.Resampling.LANCZOS)
            for img in images
        ]
        
        # Calculate grid dimensions
        rows = (len(images) + cols - 1) // cols
        img_width, img_height = target_size
        
        total_width = cols * img_width + (cols + 1) * spacing
        total_height = rows * img_height + (rows + 1) * spacing + 30 * rows  # Space for labels
        
        # Create canvas
        canvas = Image.new("RGB", (total_width, total_height), color="white")
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
        
        # Place images in grid
        for i, (img, label) in enumerate(zip(resized_images, labels)):
            row = i // cols
            col = i % cols
            
            x = spacing + col * (img_width + spacing)
            y = spacing + row * (img_height + spacing + 30)
            
            # Paste image
            canvas.paste(img, (x, y + 30))
            
            # Draw label
            draw.text(
                (x, y),
                label,
                fill="black",
                font=font
            )
        
        return canvas
    
    @staticmethod
    def create_zoom_comparison(
        original: Image.Image,
        upscaled: Image.Image,
        zoom_region: Tuple[int, int, int, int],  # (x, y, width, height)
        zoom_factor: int = 4
    ) -> Image.Image:
        """
        Create zoomed comparison of a specific region.
        
        Args:
            original: Original image
            upscaled: Upscaled image
            zoom_region: Region to zoom (x, y, width, height)
            zoom_factor: Zoom factor
            
        Returns:
            Zoom comparison image
        """
        x, y, w, h = zoom_region
        
        # Crop regions
        orig_crop = original.crop((x, y, x + w, y + h))
        upscaled_crop = upscaled.crop((
            x * (upscaled.width // original.width),
            y * (upscaled.height // original.height),
            (x + w) * (upscaled.width // original.width),
            (y + h) * (upscaled.height // original.height)
        ))
        
        # Zoom
        orig_zoom = orig_crop.resize(
            (w * zoom_factor, h * zoom_factor),
            Image.Resampling.NEAREST
        )
        upscaled_zoom = upscaled_crop.resize(
            (upscaled_crop.width * zoom_factor, upscaled_crop.height * zoom_factor),
            Image.Resampling.NEAREST
        )
        
        # Create side-by-side
        return ImageComparison.create_side_by_side(
            orig_zoom,
            upscaled_zoom,
            labels=("Original (Zoomed)", "Upscaled (Zoomed)")
        )


