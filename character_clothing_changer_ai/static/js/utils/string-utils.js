/**
 * String Utils Module
 * ===================
 * Utilities for string manipulation and operations
 */

const StringUtils = {
    /**
     * Capitalize first letter
     */
    capitalize(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    },
    
    /**
     * Title case
     */
    titleCase(str) {
        if (!str) return '';
        return str.replace(/\w\S*/g, (txt) => {
            return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
        });
    },
    
    /**
     * Camel case
     */
    camelCase(str) {
        if (!str) return '';
        return str
            .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
                return index === 0 ? word.toLowerCase() : word.toUpperCase();
            })
            .replace(/\s+/g, '');
    },
    
    /**
     * Kebab case
     */
    kebabCase(str) {
        if (!str) return '';
        return str
            .replace(/([a-z])([A-Z])/g, '$1-$2')
            .replace(/[\s_]+/g, '-')
            .toLowerCase();
    },
    
    /**
     * Snake case
     */
    snakeCase(str) {
        if (!str) return '';
        return str
            .replace(/([a-z])([A-Z])/g, '$1_$2')
            .replace(/[\s-]+/g, '_')
            .toLowerCase();
    },
    
    /**
     * Pascal case
     */
    pascalCase(str) {
        if (!str) return '';
        return str
            .replace(/(?:^\w|[A-Z]|\b\w)/g, word => word.toUpperCase())
            .replace(/\s+/g, '');
    },
    
    /**
     * Truncate string
     */
    truncate(str, maxLength, suffix = '...') {
        if (!str || str.length <= maxLength) {
            return str;
        }
        return str.substring(0, maxLength - suffix.length) + suffix;
    },
    
    /**
     * Pad string
     */
    pad(str, length, padString = ' ', padStart = true) {
        if (!str) str = '';
        const padding = padString.repeat(Math.max(0, length - str.length));
        return padStart ? padding + str : str + padding;
    },
    
    /**
     * Remove whitespace
     */
    removeWhitespace(str) {
        if (!str) return '';
        return str.replace(/\s+/g, '');
    },
    
    /**
     * Remove special characters
     */
    removeSpecialChars(str, keep = '') {
        if (!str) return '';
        const regex = new RegExp(`[^a-zA-Z0-9${keep.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}]`, 'g');
        return str.replace(regex, '');
    },
    
    /**
     * Slugify
     */
    slugify(str) {
        if (!str) return '';
        return str
            .toLowerCase()
            .trim()
            .replace(/[^\w\s-]/g, '')
            .replace(/[\s_-]+/g, '-')
            .replace(/^-+|-+$/g, '');
    },
    
    /**
     * Escape HTML
     */
    escapeHTML(str) {
        if (!str) return '';
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
     * Unescape HTML
     */
    unescapeHTML(str) {
        if (!str) return '';
        const map = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#039;': "'"
        };
        return str.replace(/&amp;|&lt;|&gt;|&quot;|&#039;/g, m => map[m]);
    },
    
    /**
     * Strip HTML tags
     */
    stripHTML(str) {
        if (!str) return '';
        return str.replace(/<[^>]*>/g, '');
    },
    
    /**
     * Extract URLs
     */
    extractURLs(str) {
        if (!str) return [];
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        return str.match(urlRegex) || [];
    },
    
    /**
     * Extract emails
     */
    extractEmails(str) {
        if (!str) return [];
        const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
        return str.match(emailRegex) || [];
    },
    
    /**
     * Mask string
     */
    mask(str, start = 0, end = 0, maskChar = '*') {
        if (!str) return '';
        if (str.length <= start + end) {
            return maskChar.repeat(str.length);
        }
        return str.substring(0, start) + 
               maskChar.repeat(str.length - start - end) + 
               str.substring(str.length - end);
    },
    
    /**
     * Reverse string
     */
    reverse(str) {
        if (!str) return '';
        return str.split('').reverse().join('');
    },
    
    /**
     * Count words
     */
    countWords(str) {
        if (!str) return 0;
        return str.trim().split(/\s+/).filter(word => word.length > 0).length;
    },
    
    /**
     * Count characters
     */
    countChars(str, includeSpaces = true) {
        if (!str) return 0;
        return includeSpaces ? str.length : str.replace(/\s/g, '').length;
    },
    
    /**
     * Check if string contains
     */
    contains(str, search, caseSensitive = false) {
        if (!str || !search) return false;
        const s = caseSensitive ? str : str.toLowerCase();
        const searchStr = caseSensitive ? search : search.toLowerCase();
        return s.includes(searchStr);
    },
    
    /**
     * Check if string starts with
     */
    startsWith(str, search, caseSensitive = false) {
        if (!str || !search) return false;
        const s = caseSensitive ? str : str.toLowerCase();
        const searchStr = caseSensitive ? search : search.toLowerCase();
        return s.startsWith(searchStr);
    },
    
    /**
     * Check if string ends with
     */
    endsWith(str, search, caseSensitive = false) {
        if (!str || !search) return false;
        const s = caseSensitive ? str : str.toLowerCase();
        const searchStr = caseSensitive ? search : search.toLowerCase();
        return s.endsWith(searchStr);
    },
    
    /**
     * Generate random string
     */
    random(length = 10, chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789') {
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StringUtils;
}

