/**
 * Security Manager Module
 * ======================
 * Manages security features like input sanitization, XSS prevention, and CSRF protection
 */

const SecurityManager = {
    /**
     * Sanitize HTML string
     */
    sanitizeHTML(str) {
        if (typeof str !== 'string') return str;
        
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },
    
    /**
     * Escape HTML special characters
     */
    escapeHTML(str) {
        if (typeof str !== 'string') return str;
        
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return str.replace(/[&<>"']/g, m => map[m]);
    },
    
    /**
     * Validate and sanitize user input
     */
    sanitizeInput(input, type = 'text') {
        if (input === null || input === undefined) return '';
        
        let sanitized = String(input).trim();
        
        switch (type) {
            case 'text':
                // Remove potentially dangerous characters
                sanitized = sanitized.replace(/[<>]/g, '');
                break;
            case 'url':
                // Validate URL
                try {
                    new URL(sanitized);
                } catch {
                    sanitized = '';
                }
                break;
            case 'email':
                // Basic email validation
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(sanitized)) {
                    sanitized = '';
                }
                break;
            case 'number':
                // Validate number
                const num = parseFloat(sanitized);
                sanitized = isNaN(num) ? '' : String(num);
                break;
        }
        
        return sanitized;
    },
    
    /**
     * Generate CSRF token
     */
    generateCSRFToken() {
        const array = new Uint8Array(32);
        crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    },
    
    /**
     * Validate CSRF token
     */
    validateCSRFToken(token, storedToken) {
        return token && storedToken && token === storedToken;
    },
    
    /**
     * Check if URL is safe
     */
    isSafeURL(url) {
        try {
            const urlObj = new URL(url);
            // Allow only http, https, and data URLs
            return ['http:', 'https:', 'data:'].includes(urlObj.protocol);
        } catch {
            return false;
        }
    },
    
    /**
     * Validate file type
     */
    validateFileType(file, allowedTypes) {
        if (!file || !file.type) return false;
        return allowedTypes.includes(file.type);
    },
    
    /**
     * Validate file size
     */
    validateFileSize(file, maxSize) {
        if (!file || !file.size) return false;
        return file.size <= maxSize;
    },
    
    /**
     * Create secure download link
     */
    createSecureDownload(data, filename, mimeType = 'application/octet-stream') {
        const blob = new Blob([data], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = this.sanitizeInput(filename, 'text');
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    },
    
    /**
     * Check Content Security Policy compliance
     */
    checkCSP() {
        // Check if CSP is enabled
        const metaTags = document.querySelectorAll('meta[http-equiv="Content-Security-Policy"]');
        return metaTags.length > 0;
    },
    
    /**
     * Initialize security features
     */
    init() {
        // Set up security headers if possible
        if (typeof Logger !== 'undefined') {
            Logger.info('Security manager initialized');
        }
        
        // Warn if CSP is not set
        if (!this.checkCSP() && typeof Logger !== 'undefined') {
            Logger.warn('Content Security Policy not detected. Consider adding CSP headers.');
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    SecurityManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SecurityManager;
}

