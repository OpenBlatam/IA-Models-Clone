"""
Image generation evaluation metrics.

This module provides comprehensive evaluation metrics for image generation models
including FID, IS, LPIPS, CLIP score, and custom image quality metrics.
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def calculate_fid_score(real_images: torch.Tensor, generated_images: torch.Tensor, 
                       device: str = "cuda") -> float:
    """Calculate Fréchet Inception Distance (FID) score."""
    if not TORCHMETRICS_AVAILABLE:
        return float('inf')
    
    try:
        fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        
        # Update with real images
        fid.update(real_images, real=True)
        
        # Update with generated images
        fid.update(generated_images, real=False)
        
        # Compute FID score
        fid_score = fid.compute().item()
        
        return fid_score
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return float('inf')


def calculate_inception_score(generated_images: torch.Tensor, 
                            device: str = "cuda") -> Tuple[float, float]:
    """Calculate Inception Score (IS) and standard deviation."""
    if not TORCHMETRICS_AVAILABLE:
        return 0.0, 0.0
    
    try:
        inception = InceptionScore(normalize=True).to(device)
        
        # Update with generated images
        inception.update(generated_images)
        
        # Compute IS score and std
        is_score, is_std = inception.compute()
        
        return is_score.item(), is_std.item()
    except Exception as e:
        print(f"Error calculating Inception Score: {e}")
        return 0.0, 0.0


def calculate_lpips_score(real_images: torch.Tensor, generated_images: torch.Tensor,
                         device: str = "cuda") -> float:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) score."""
    if not TORCHMETRICS_AVAILABLE:
        return float('inf')
    
    try:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
        
        # Calculate LPIPS score
        lpips_score = lpips(real_images, generated_images).item()
        
        return lpips_score
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return float('inf')


def calculate_clip_score(generated_images: torch.Tensor, text_prompts: List[str],
                        device: str = "cuda") -> Dict[str, float]:
    """Calculate CLIP score for image-text similarity."""
    if not CLIP_AVAILABLE:
        return {"clip_score": 0.0, "clip_score_std": 0.0}
    
    try:
        # Load CLIP model
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Preprocess images
        image_input = torch.stack([preprocess(Image.fromarray(img.cpu().numpy())) 
                                 for img in generated_images]).to(device)
        
        # Encode text prompts
        text_input = clip.tokenize(text_prompts).to(device)
        
        # Calculate similarity scores
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            
            # Calculate cosine similarity
            similarity_scores = (image_features * text_features).sum(dim=1)
        
        # Convert to numpy for calculations
        scores = similarity_scores.cpu().numpy()
        
        return {
            "clip_score": float(np.mean(scores)),
            "clip_score_std": float(np.std(scores))
        }
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return {"clip_score": 0.0, "clip_score_std": 0.0}


def calculate_image_quality_metrics(generated_images: torch.Tensor) -> Dict[str, float]:
    """Calculate custom image quality metrics."""
    metrics = {}
    
    # Convert to numpy for calculations
    if isinstance(generated_images, torch.Tensor):
        images_np = generated_images.cpu().numpy()
    else:
        images_np = generated_images
    
    # Brightness metrics
    brightness_values = []
    for img in images_np:
        if len(img.shape) == 3:
            # RGB image
            brightness = np.mean(img)
        else:
            # Grayscale image
            brightness = np.mean(img)
        brightness_values.append(brightness)
    
    metrics["avg_brightness"] = float(np.mean(brightness_values))
    metrics["brightness_std"] = float(np.std(brightness_values))
    
    # Contrast metrics
    contrast_values = []
    for img in images_np:
        if len(img.shape) == 3:
            # Calculate contrast for each channel
            channel_contrasts = []
            for channel in range(img.shape[0]):
                channel_data = img[channel]
                channel_contrast = np.std(channel_data)
                channel_contrasts.append(channel_contrast)
            contrast = np.mean(channel_contrasts)
        else:
            contrast = np.std(img)
        contrast_values.append(contrast)
    
    metrics["avg_contrast"] = float(np.mean(contrast_values))
    metrics["contrast_std"] = float(np.std(contrast_values))
    
    # Sharpness metrics (using Laplacian variance)
    sharpness_values = []
    for img in images_np:
        if len(img.shape) == 3:
            # Convert to grayscale for sharpness calculation
            gray_img = np.mean(img, axis=0)
        else:
            gray_img = img
        
        # Calculate Laplacian variance as sharpness measure
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        sharpness = np.var(F.conv2d(torch.tensor(gray_img).unsqueeze(0).unsqueeze(0), 
                           torch.tensor(laplacian).float().unsqueeze(0).unsqueeze(0)).numpy())
        sharpness_values.append(sharpness)
    
    metrics["avg_sharpness"] = float(np.mean(sharpness_values))
    metrics["sharpness_std"] = float(np.std(sharpness_values))
    
    # Color diversity (for RGB images)
    if len(images_np[0].shape) == 3:
        color_diversity_values = []
        for img in images_np:
            # Calculate color histogram diversity
            hist_r = np.histogram(img[0], bins=16, range=(0, 1))[0]
            hist_g = np.histogram(img[1], bins=16, range=(0, 1))[0]
            hist_b = np.histogram(img[2], bins=16, range=(0, 1))[0]
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            # Calculate entropy as diversity measure
            entropy_r = -np.sum(hist_r * np.log2(hist_r + 1e-10))
            entropy_g = -np.sum(hist_g * np.log2(hist_g + 1e-10))
            entropy_b = -np.sum(hist_b * np.log2(hist_b + 1e-10))
            
            avg_entropy = (entropy_r + entropy_g + entropy_b) / 3
            color_diversity_values.append(avg_entropy)
        
        metrics["avg_color_diversity"] = float(np.mean(color_diversity_values))
        metrics["color_diversity_std"] = float(np.std(color_diversity_values))
    
    return metrics


def calculate_structural_similarity(real_images: torch.Tensor, 
                                  generated_images: torch.Tensor) -> Dict[str, float]:
    """Calculate structural similarity metrics."""
    metrics = {}
    
    # Convert to numpy
    if isinstance(real_images, torch.Tensor):
        real_np = real_images.cpu().numpy()
    else:
        real_np = real_images
    
    if isinstance(generated_images, torch.Tensor):
        gen_np = generated_images.cpu().numpy()
    else:
        gen_np = generated_images
    
    # MSE (Mean Squared Error)
    mse_values = []
    for real_img, gen_img in zip(real_np, gen_np):
        mse = np.mean((real_img - gen_img) ** 2)
        mse_values.append(mse)
    
    metrics["mse"] = float(np.mean(mse_values))
    metrics["mse_std"] = float(np.std(mse_values))
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr_values = []
    for real_img, gen_img in zip(real_np, gen_np):
        mse = np.mean((real_img - gen_img) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr_values.append(psnr)
    
    # Filter out infinite values
    psnr_values = [p for p in psnr_values if p != float('inf')]
    if psnr_values:
        metrics["psnr"] = float(np.mean(psnr_values))
        metrics["psnr_std"] = float(np.std(psnr_values))
    else:
        metrics["psnr"] = 0.0
        metrics["psnr_std"] = 0.0
    
    # SSIM approximation (simplified version)
    ssim_values = []
    for real_img, gen_img in zip(real_np, gen_np):
        # Simplified SSIM calculation
        mu_real = np.mean(real_img)
        mu_gen = np.mean(gen_img)
        sigma_real = np.std(real_img)
        sigma_gen = np.std(gen_img)
        
        # Cross-covariance
        cross_cov = np.mean((real_img - mu_real) * (gen_img - mu_gen))
        
        # SSIM formula
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        numerator = (2 * mu_real * mu_gen + c1) * (2 * cross_cov + c2)
        denominator = (mu_real ** 2 + mu_gen ** 2 + c1) * (sigma_real ** 2 + sigma_gen ** 2 + c2)
        
        ssim = numerator / denominator if denominator != 0 else 0
        ssim_values.append(ssim)
    
    metrics["ssim"] = float(np.mean(ssim_values))
    metrics["ssim_std"] = float(np.std(ssim_values))
    
    return metrics


def evaluate_diffusion_model(model, data_loader: DataLoader, device: str,
                           real_images: Optional[torch.Tensor] = None,
                           generated_images: Optional[torch.Tensor] = None,
                           text_prompts: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of diffusion models.
    
    Args:
        model: The diffusion model to evaluate
        data_loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        real_images: Optional tensor of real images
        generated_images: Optional tensor of generated images
        text_prompts: Optional list of text prompts for CLIP score
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    metrics = {}
    
    # Calculate image-based metrics if images are provided
    if real_images is not None and generated_images is not None:
        # Ensure images are in the right format
        if real_images.dim() == 3:
            real_images = real_images.unsqueeze(0)
        if generated_images.dim() == 3:
            generated_images = generated_images.unsqueeze(0)
        
        # FID score
        fid_score = calculate_fid_score(real_images, generated_images, device)
        metrics["fid_score"] = fid_score
        
        # LPIPS score
        lpips_score = calculate_lpips_score(real_images, generated_images, device)
        metrics["lpips_score"] = lpips_score
        
        # Structural similarity metrics
        struct_metrics = calculate_structural_similarity(real_images, generated_images)
        metrics.update(struct_metrics)
    
    # Calculate generation quality metrics
    if generated_images is not None:
        # Inception Score
        is_score, is_std = calculate_inception_score(generated_images, device)
        metrics["inception_score"] = is_score
        metrics["inception_score_std"] = is_std
        
        # Image quality metrics
        quality_metrics = calculate_image_quality_metrics(generated_images)
        metrics.update(quality_metrics)
        
        # CLIP score if text prompts are provided
        if text_prompts and len(text_prompts) == len(generated_images):
            clip_metrics = calculate_clip_score(generated_images, text_prompts, device)
            metrics.update(clip_metrics)
    
    return metrics


def evaluate_image_generation_batch(real_images: torch.Tensor,
                                   generated_images: torch.Tensor,
                                   text_prompts: Optional[List[str]] = None,
                                   device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate a batch of generated images against real images.
    
    Args:
        real_images: Tensor of real images
        generated_images: Tensor of generated images
        text_prompts: Optional list of text prompts for CLIP score
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    if real_images.shape != generated_images.shape:
        raise ValueError("Real and generated images must have the same shape")
    
    metrics = {}
    
    # FID score
    fid_score = calculate_fid_score(real_images, generated_images, device)
    metrics["fid_score"] = fid_score
    
    # LPIPS score
    lpips_score = calculate_lpips_score(real_images, generated_images, device)
    metrics["lpips_score"] = lpips_score
    
    # Inception Score
    is_score, is_std = calculate_inception_score(generated_images, device)
    metrics["inception_score"] = is_score
    metrics["inception_score_std"] = is_std
    
    # Structural similarity metrics
    struct_metrics = calculate_structural_similarity(real_images, generated_images)
    metrics.update(struct_metrics)
    
    # Image quality metrics
    quality_metrics = calculate_image_quality_metrics(generated_images)
    metrics.update(quality_metrics)
    
    # CLIP score if text prompts are provided
    if text_prompts and len(text_prompts) == len(generated_images):
        clip_metrics = calculate_clip_score(generated_images, text_prompts, device)
        metrics.update(clip_metrics)
    
    return metrics
