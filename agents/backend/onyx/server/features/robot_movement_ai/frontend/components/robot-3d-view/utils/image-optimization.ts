/**
 * Image optimization utilities
 * @module robot-3d-view/utils/image-optimization
 */

/**
 * Image optimization options
 */
export interface ImageOptimizationOptions {
  maxWidth?: number;
  maxHeight?: number;
  quality?: number;
  format?: 'image/jpeg' | 'image/png' | 'image/webp';
  progressive?: boolean;
}

/**
 * Image Optimization Manager class
 */
export class ImageOptimizationManager {
  /**
   * Optimizes an image file
   */
  async optimizeImage(
    file: File,
    options: ImageOptimizationOptions = {}
  ): Promise<Blob> {
    const {
      maxWidth = 1920,
      maxHeight = 1080,
      quality = 0.8,
      format = 'image/jpeg',
    } = options;

    return new Promise((resolve, reject) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        reject(new Error('Canvas context not available'));
        return;
      }

      img.onload = () => {
        // Calculate new dimensions
        let { width, height } = img;
        const ratio = Math.min(maxWidth / width, maxHeight / height);

        if (ratio < 1) {
          width *= ratio;
          height *= ratio;
        }

        // Set canvas dimensions
        canvas.width = width;
        canvas.height = height;

        // Draw and compress
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to create blob'));
            }
          },
          format,
          quality
        );
      };

      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };

      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Compresses an image
   */
  async compressImage(file: File, quality = 0.8): Promise<Blob> {
    return this.optimizeImage(file, { quality });
  }

  /**
   * Resizes an image
   */
  async resizeImage(
    file: File,
    maxWidth: number,
    maxHeight: number
  ): Promise<Blob> {
    return this.optimizeImage(file, { maxWidth, maxHeight });
  }

  /**
   * Converts image format
   */
  async convertImageFormat(
    file: File,
    format: 'image/jpeg' | 'image/png' | 'image/webp'
  ): Promise<Blob> {
    return this.optimizeImage(file, { format });
  }

  /**
   * Gets image dimensions
   */
  getImageDimensions(file: File): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        resolve({ width: img.width, height: img.height });
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Creates a thumbnail
   */
  async createThumbnail(
    file: File,
    size = 150
  ): Promise<Blob> {
    return this.optimizeImage(file, {
      maxWidth: size,
      maxHeight: size,
      quality: 0.7,
    });
  }

  /**
   * Checks if WebP is supported
   */
  isWebPSupported(): boolean {
    const canvas = document.createElement('canvas');
    return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
  }

  /**
   * Gets optimal format
   */
  getOptimalFormat(): 'image/jpeg' | 'image/png' | 'image/webp' {
    return this.isWebPSupported() ? 'image/webp' : 'image/jpeg';
  }
}

/**
 * Global image optimization manager instance
 */
export const imageOptimizationManager = new ImageOptimizationManager();



