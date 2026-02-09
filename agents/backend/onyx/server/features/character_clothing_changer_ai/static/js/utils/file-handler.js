/**
 * File Handler Module
 * ===================
 * Handles file operations and preview
 */

const FileHandler = {
    /**
     * Read file as data URL
     */
    readAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    /**
     * Setup file preview
     */
    setupPreview(file, previewElement, labelElement) {
        return this.readAsDataURL(file).then(dataURL => {
            if (previewElement) {
                previewElement.src = dataURL;
                previewElement.style.display = 'block';
            }
            
            if (labelElement) {
                labelElement.classList.add('has-file');
                const icon = labelElement.querySelector('div:first-child');
                const name = labelElement.querySelector('div:nth-child(2)');
                if (icon) icon.textContent = '✅';
                if (name) name.textContent = file.name;
            }
            
            return dataURL;
        });
    },

    /**
     * Validate file
     */
    validateFile(file, options = {}) {
        const {
            maxSize = 10 * 1024 * 1024, // 10MB default
            allowedTypes = ['image/jpeg', 'image/png', 'image/webp'],
            allowedExtensions = ['.jpg', '.jpeg', '.png', '.webp']
        } = options;
        
        const errors = [];
        
        if (!file) {
            errors.push('No se seleccionó ningún archivo');
            return { valid: false, errors };
        }
        
        if (file.size > maxSize) {
            errors.push(`El archivo es demasiado grande (máximo ${maxSize / 1024 / 1024}MB)`);
        }
        
        if (!allowedTypes.includes(file.type)) {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowedExtensions.includes(extension)) {
                errors.push(`Tipo de archivo no permitido. Use: ${allowedExtensions.join(', ')}`);
            }
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    },

    /**
     * Get file info
     */
    getFileInfo(file) {
        if (!file) return null;
        
        return {
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified,
            sizeFormatted: this.formatFileSize(file.size)
        };
    },

    /**
     * Format file size (delegates to FormatUtils)
     */
    formatFileSize(bytes) {
        if (typeof FormatUtils !== 'undefined') {
            return FormatUtils.formatBytes(bytes);
        }
        // Fallback implementation
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FileHandler;
}

