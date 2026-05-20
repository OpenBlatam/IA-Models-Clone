/**
 * Security Manager Module
 * =======================
 * Security utilities and XSS prevention
 */

const SecurityManager = {
    /**
     * Sanitize HTML to prevent XSS
     */
    sanitizeHTML(html) {
        const div = document.createElement('div');
        div.textContent = html;
        return div.innerHTML;
    },
    
    /**
     * Escape HTML entities
     */
    escapeHTML(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    },
    
    /**
     * Validate URL
     */
    isValidURL(url) {
        try {
            const parsed = new URL(url);
            // Only allow http, https, and data URLs
            return ['http:', 'https:', 'data:'].includes(parsed.protocol);
        } catch {
            return false;
        }
    },
    
    /**
     * Sanitize URL
     */
    sanitizeURL(url) {
        if (!this.isValidURL(url)) {
            return 'about:blank';
        }
        return url;
    },
    
    /**
     * Validate and sanitize user input
     */
    sanitizeInput(input, type = 'text') {
        if (typeof input !== 'string') {
            return '';
        }
        
        switch (type) {
            case 'html':
                return this.sanitizeHTML(input);
            case 'url':
                return this.sanitizeURL(input);
            case 'text':
            default:
                return this.escapeHTML(input);
        }
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
        return token === storedToken;
    },
    
    /**
     * Check if content is safe
     */
    isContentSafe(content) {
        // Check for common XSS patterns
        const dangerousPatterns = [
            /<script/i,
            /javascript:/i,
            /on\w+\s*=/i,
            /<iframe/i,
            /<object/i,
            /<embed/i
        ];
        
        return !dangerousPatterns.some(pattern => pattern.test(content));
    },
    
    /**
     * Sanitize object recursively
     */
    sanitizeObject(obj) {
        if (typeof obj === 'string') {
            return this.escapeHTML(obj);
        }
        
        if (Array.isArray(obj)) {
            return obj.map(item => this.sanitizeObject(item));
        }
        
        if (obj && typeof obj === 'object') {
            const sanitized = {};
            for (const [key, value] of Object.entries(obj)) {
                sanitized[this.escapeHTML(key)] = this.sanitizeObject(value);
            }
            return sanitized;
        }
        
        return obj;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SecurityManager;
}

