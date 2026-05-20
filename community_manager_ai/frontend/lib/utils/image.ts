/**
 * Image Utility Functions
 * Utilities for optimizing and handling images
 */

/**
 * Generates optimized image URL with Next.js Image optimization
 * @param src - Image source URL
 * @param width - Image width
 * @param height - Optional image height
 * @param quality - Image quality (1-100, default: 75)
 * @returns Optimized image URL
 */
export const getOptimizedImageUrl = (
  src: string,
  width: number,
  height?: number,
  quality: number = 75
): string => {
  if (!src) return '';
  
  // If it's already an absolute URL, return as is
  if (src.startsWith('http://') || src.startsWith('https://')) {
    return src;
  }

  // For Next.js Image optimization
  const params = new URLSearchParams({
    url: src,
    w: width.toString(),
    q: quality.toString(),
  });

  if (height) {
    params.set('h', height.toString());
  }

  return `/api/image?${params.toString()}`;
};

/**
 * Generates responsive image srcset
 * @param src - Base image source
 * @param sizes - Array of sizes in pixels
 * @returns Srcset string
 */
export const generateSrcSet = (src: string, sizes: number[]): string => {
  return sizes
    .map((size) => `${getOptimizedImageUrl(src, size)} ${size}w`)
    .join(', ');
};

/**
 * Gets image dimensions from URL or file
 * @param src - Image source
 * @returns Promise with image dimensions
 */
export const getImageDimensions = (src: string): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = reject;
    img.src = src;
  });
};

/**
 * Checks if an image URL is valid
 * @param url - Image URL to validate
 * @returns Promise that resolves to true if image is valid
 */
export const validateImageUrl = (url: string): Promise<boolean> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = url;
  });
};

/**
 * Formats file size to human-readable format
 * @param bytes - File size in bytes
 * @returns Formatted file size string
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

/**
 * Gets image format from URL
 * @param url - Image URL
 * @returns Image format (webp, jpeg, png, etc.)
 */
export const getImageFormat = (url: string): string => {
  const extension = url.split('.').pop()?.toLowerCase() || '';
  const formats: Record<string, string> = {
    jpg: 'jpeg',
    jpeg: 'jpeg',
    png: 'png',
    webp: 'webp',
    gif: 'gif',
    svg: 'svg',
    avif: 'avif',
  };
  return formats[extension] || 'jpeg';
};


