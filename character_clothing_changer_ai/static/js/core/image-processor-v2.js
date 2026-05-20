/**
 * Image Processor V2 Module
 * ========================
 * Advanced image processing with Web Workers and Canvas API
 */

const ImageProcessorV2 = {
    /**
     * Worker pool
     */
    workerPool: null,
    
    /**
     * Processing queue
     */
    queue: [],
    
    /**
     * Initialize image processor
     */
    async init() {
        // Create worker pool if WorkerManager is available
        if (typeof WorkerManager !== 'undefined') {
            try {
                this.workerPool = WorkerManager.createPool('image-processing', '/js/workers/image-processor-worker.js', 2);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.warn('Worker pool not available, using main thread', error);
                }
            }
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Image processor V2 initialized');
        }
    },
    
    /**
     * Resize image
     */
    async resize(image, width, height, options = {}) {
        const {
            maintainAspectRatio = true,
            quality = 0.92,
            format = 'image/jpeg'
        } = options;
        
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Calculate dimensions
            let targetWidth = width;
            let targetHeight = height;
            
            if (maintainAspectRatio) {
                const aspectRatio = image.width / image.height;
                if (width && !height) {
                    targetHeight = width / aspectRatio;
                } else if (height && !width) {
                    targetWidth = height * aspectRatio;
                } else {
                    const imageAspect = image.width / image.height;
                    const targetAspect = width / height;
                    
                    if (imageAspect > targetAspect) {
                        targetHeight = width / imageAspect;
                    } else {
                        targetWidth = height * imageAspect;
                    }
                }
            }
            
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            
            // Draw image
            ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
            
            // Convert to blob
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to resize image'));
                }
            }, format, quality);
        });
    },
    
    /**
     * Compress image
     */
    async compress(image, quality = 0.8, format = 'image/jpeg') {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = image.width;
            canvas.height = image.height;
            
            ctx.drawImage(image, 0, 0);
            
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to compress image'));
                }
            }, format, quality);
        });
    },
    
    /**
     * Crop image
     */
    async crop(image, x, y, width, height) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = width;
            canvas.height = height;
            
            ctx.drawImage(image, x, y, width, height, 0, 0, width, height);
            
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to crop image'));
                }
            });
        });
    },
    
    /**
     * Apply filter
     */
    async applyFilter(image, filterType, options = {}) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = image.width;
            canvas.height = image.height;
            
            ctx.drawImage(image, 0, 0);
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Apply filter
            switch (filterType) {
                case 'grayscale':
                    for (let i = 0; i < data.length; i += 4) {
                        const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
                        data[i] = gray;
                        data[i + 1] = gray;
                        data[i + 2] = gray;
                    }
                    break;
                    
                case 'sepia':
                    for (let i = 0; i < data.length; i += 4) {
                        const r = data[i];
                        const g = data[i + 1];
                        const b = data[i + 2];
                        data[i] = Math.min(255, (r * 0.393) + (g * 0.769) + (b * 0.189));
                        data[i + 1] = Math.min(255, (r * 0.349) + (g * 0.686) + (b * 0.168));
                        data[i + 2] = Math.min(255, (r * 0.272) + (g * 0.534) + (b * 0.131));
                    }
                    break;
                    
                case 'brightness':
                    const brightness = options.value || 0;
                    for (let i = 0; i < data.length; i += 4) {
                        data[i] = Math.min(255, Math.max(0, data[i] + brightness));
                        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + brightness));
                        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + brightness));
                    }
                    break;
                    
                case 'contrast':
                    const contrast = options.value || 1;
                    const factor = (259 * (contrast * 255 + 255)) / (255 * (259 - contrast * 255));
                    for (let i = 0; i < data.length; i += 4) {
                        data[i] = Math.min(255, Math.max(0, factor * (data[i] - 128) + 128));
                        data[i + 1] = Math.min(255, Math.max(0, factor * (data[i + 1] - 128) + 128));
                        data[i + 2] = Math.min(255, Math.max(0, factor * (data[i + 2] - 128) + 128));
                    }
                    break;
            }
            
            ctx.putImageData(imageData, 0, 0);
            
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to apply filter'));
                }
            });
        });
    },
    
    /**
     * Convert to base64
     */
    async toBase64(image, format = 'image/jpeg', quality = 0.92) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = image.width;
            canvas.height = image.height;
            
            ctx.drawImage(image, 0, 0);
            
            try {
                const base64 = canvas.toDataURL(format, quality);
                resolve(base64);
            } catch (error) {
                reject(error);
            }
        });
    },
    
    /**
     * Load image from URL
     */
    loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    },
    
    /**
     * Load image from blob
     */
    loadImageFromBlob(blob) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const url = URL.createObjectURL(blob);
            
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to load image from blob'));
            };
            img.src = url;
        });
    },
    
    /**
     * Get image info
     */
    getImageInfo(image) {
        return {
            width: image.width,
            height: image.height,
            aspectRatio: image.width / image.height,
            size: image.width * image.height
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImageProcessorV2;
}

