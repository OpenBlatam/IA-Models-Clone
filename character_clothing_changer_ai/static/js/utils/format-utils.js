/**
 * Format Utils Module
 * ===================
 * Utilities for formatting data (numbers, currency, bytes, etc.)
 */

const FormatUtils = {
    /**
     * Format number
     */
    formatNumber(number, decimals = 2, locale = 'es-ES') {
        if (number === null || number === undefined || isNaN(number)) {
            return '0';
        }
        
        return new Intl.NumberFormat(locale, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(number);
    },
    
    /**
     * Format currency
     */
    formatCurrency(amount, currency = 'USD', locale = 'es-ES') {
        if (amount === null || amount === undefined || isNaN(amount)) {
            return '0.00';
        }
        
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currency
        }).format(amount);
    },
    
    /**
     * Format bytes
     */
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },
    
    /**
     * Format percentage
     */
    formatPercentage(value, decimals = 2) {
        if (value === null || value === undefined || isNaN(value)) {
            return '0%';
        }
        
        return `${value.toFixed(decimals)}%`;
    },
    
    /**
     * Format duration
     */
    formatDuration(milliseconds) {
        if (milliseconds < 1000) {
            return `${milliseconds}ms`;
        }
        
        const seconds = Math.floor(milliseconds / 1000);
        if (seconds < 60) {
            return `${seconds}s`;
        }
        
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        if (minutes < 60) {
            return `${minutes}m ${remainingSeconds}s`;
        }
        
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = minutes % 60;
        return `${hours}h ${remainingMinutes}m`;
    },
    
    /**
     * Format file size
     */
    formatFileSize(bytes) {
        return this.formatBytes(bytes);
    },
    
    /**
     * Format phone number
     */
    formatPhoneNumber(phone, format = 'default') {
        if (!phone) return '';
        
        const cleaned = phone.replace(/\D/g, '');
        
        switch (format) {
            case 'us':
                if (cleaned.length === 10) {
                    return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
                }
                break;
            case 'international':
                if (cleaned.length > 0) {
                    return `+${cleaned}`;
                }
                break;
            default:
                return cleaned;
        }
        
        return phone;
    },
    
    /**
     * Format credit card
     */
    formatCreditCard(cardNumber) {
        if (!cardNumber) return '';
        
        const cleaned = cardNumber.replace(/\D/g, '');
        const formatted = cleaned.match(/.{1,4}/g)?.join(' ') || cleaned;
        
        return formatted;
    },
    
    /**
     * Truncate text
     */
    truncate(text, maxLength = 50, suffix = '...') {
        if (!text || text.length <= maxLength) {
            return text;
        }
        
        return text.substring(0, maxLength - suffix.length) + suffix;
    },
    
    /**
     * Capitalize first letter
     */
    capitalize(text) {
        if (!text) return '';
        return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
    },
    
    /**
     * Title case
     */
    titleCase(text) {
        if (!text) return '';
        return text.replace(/\w\S*/g, (txt) => {
            return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
        });
    },
    
    /**
     * Format slug
     */
    formatSlug(text) {
        if (!text) return '';
        return text
            .toLowerCase()
            .trim()
            .replace(/[^\w\s-]/g, '')
            .replace(/[\s_-]+/g, '-')
            .replace(/^-+|-+$/g, '');
    },
    
    /**
     * Format initials
     */
    formatInitials(name, maxInitials = 2) {
        if (!name) return '';
        
        const words = name.trim().split(/\s+/);
        const initials = words
            .slice(0, maxInitials)
            .map(word => word.charAt(0).toUpperCase())
            .join('');
        
        return initials;
    },
    
    /**
     * Format mask
     */
    formatMask(value, mask) {
        if (!value || !mask) return value;
        
        let maskedValue = '';
        let valueIndex = 0;
        
        for (let i = 0; i < mask.length && valueIndex < value.length; i++) {
            if (mask[i] === '#') {
                maskedValue += value[valueIndex++];
            } else {
                maskedValue += mask[i];
            }
        }
        
        return maskedValue;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FormatUtils;
}

