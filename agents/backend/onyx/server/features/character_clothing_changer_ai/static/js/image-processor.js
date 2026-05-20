/**
 * Image Processor Module
 * ======================
 * Client-side image processing utilities
 */

const ImageProcessor = {
    /**
     * Resize image
     */
    async resize(image, width, height, quality = 0.9) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0, width, height);
            
            canvas.toBlob(resolve, 'image/jpeg', quality);
        });
    },
    
    /**
     * Compress image
     */
    async compress(image, maxSizeKB = 500, quality = 0.8) {
        let compressed = image;
        let currentQuality = quality;
        
        while (currentQuality > 0.1) {
            const blob = await this.imageToBlob(compressed, currentQuality);
            const sizeKB = blob.size / 1024;
            
            if (sizeKB <= maxSizeKB) {
                return blob;
            }
            
            currentQuality -= 0.1;
        }
        
        return this.imageToBlob(compressed, 0.1);
    },
    
    /**
     * Convert image to blob
     */
    async imageToBlob(image, quality = 0.9) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            canvas.width = image.width || image.naturalWidth;
            canvas.height = image.height || image.naturalHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            
            canvas.toBlob(resolve, 'image/jpeg', quality);
        });
    },
    
    /**
     * Crop image
     */
    async crop(image, x, y, width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, x, y, width, height, 0, 0, width, height);
        
        return new Promise((resolve) => {
            canvas.toBlob(resolve, 'image/jpeg', 0.9);
        });
    },
    
    /**
     * Get image data URL
     */
    async toDataURL(image, format = 'image/jpeg', quality = 0.9) {
        const canvas = document.createElement('canvas');
        canvas.width = image.width || image.naturalWidth;
        canvas.height = image.height || image.naturalHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        
        return canvas.toDataURL(format, quality);
    },
    
    /**
     * Load image from URL
     */
    async loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
            img.src = url;
        });
    },
    
    /**
     * Get image dimensions
     */
    getDimensions(image) {
        return {
            width: image.width || image.naturalWidth,
            height: image.height || image.naturalHeight
        };
    },
    
    /**
     * Calculate aspect ratio
     */
    getAspectRatio(image) {
        const dims = this.getDimensions(image);
        return dims.width / dims.height;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageProcessor;
}

