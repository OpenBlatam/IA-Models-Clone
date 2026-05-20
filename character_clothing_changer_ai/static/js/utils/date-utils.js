/**
 * Date Utils Module
 * =================
 * Utilities for date manipulation and formatting
 */

const DateUtils = {
    /**
     * Format date
     */
    format(date, format = 'YYYY-MM-DD HH:mm:ss') {
        if (!date) return '';
        
        const d = new Date(date);
        if (isNaN(d.getTime())) return '';
        
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hours = String(d.getHours()).padStart(2, '0');
        const minutes = String(d.getMinutes()).padStart(2, '0');
        const seconds = String(d.getSeconds()).padStart(2, '0');
        
        return format
            .replace('YYYY', year)
            .replace('MM', month)
            .replace('DD', day)
            .replace('HH', hours)
            .replace('mm', minutes)
            .replace('ss', seconds);
    },
    
    /**
     * Format relative time
     */
    formatRelative(date) {
        if (!date) return '';
        
        const d = new Date(date);
        if (isNaN(d.getTime())) return '';
        
        const now = new Date();
        const diff = now - d;
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        const weeks = Math.floor(days / 7);
        const months = Math.floor(days / 30);
        const years = Math.floor(days / 365);
        
        if (seconds < 60) return 'hace un momento';
        if (minutes < 60) return `hace ${minutes} ${minutes === 1 ? 'minuto' : 'minutos'}`;
        if (hours < 24) return `hace ${hours} ${hours === 1 ? 'hora' : 'horas'}`;
        if (days < 7) return `hace ${days} ${days === 1 ? 'día' : 'días'}`;
        if (weeks < 4) return `hace ${weeks} ${weeks === 1 ? 'semana' : 'semanas'}`;
        if (months < 12) return `hace ${months} ${months === 1 ? 'mes' : 'meses'}`;
        return `hace ${years} ${years === 1 ? 'año' : 'años'}`;
    },
    
    /**
     * Get start of day
     */
    startOfDay(date = new Date()) {
        const d = new Date(date);
        d.setHours(0, 0, 0, 0);
        return d;
    },
    
    /**
     * Get end of day
     */
    endOfDay(date = new Date()) {
        const d = new Date(date);
        d.setHours(23, 59, 59, 999);
        return d;
    },
    
    /**
     * Add time to date
     */
    add(date, amount, unit = 'days') {
        const d = new Date(date);
        
        switch (unit) {
            case 'years':
                d.setFullYear(d.getFullYear() + amount);
                break;
            case 'months':
                d.setMonth(d.getMonth() + amount);
                break;
            case 'weeks':
                d.setDate(d.getDate() + (amount * 7));
                break;
            case 'days':
                d.setDate(d.getDate() + amount);
                break;
            case 'hours':
                d.setHours(d.getHours() + amount);
                break;
            case 'minutes':
                d.setMinutes(d.getMinutes() + amount);
                break;
            case 'seconds':
                d.setSeconds(d.getSeconds() + amount);
                break;
        }
        
        return d;
    },
    
    /**
     * Get difference between dates
     */
    diff(date1, date2, unit = 'days') {
        const d1 = new Date(date1);
        const d2 = new Date(date2);
        const diff = Math.abs(d2 - d1);
        
        switch (unit) {
            case 'years':
                return Math.floor(diff / (365 * 24 * 60 * 60 * 1000));
            case 'months':
                return Math.floor(diff / (30 * 24 * 60 * 60 * 1000));
            case 'weeks':
                return Math.floor(diff / (7 * 24 * 60 * 60 * 1000));
            case 'days':
                return Math.floor(diff / (24 * 60 * 60 * 1000));
            case 'hours':
                return Math.floor(diff / (60 * 60 * 1000));
            case 'minutes':
                return Math.floor(diff / (60 * 1000));
            case 'seconds':
                return Math.floor(diff / 1000);
            default:
                return diff;
        }
    },
    
    /**
     * Check if date is today
     */
    isToday(date) {
        const d = new Date(date);
        const today = new Date();
        return d.toDateString() === today.toDateString();
    },
    
    /**
     * Check if date is in past
     */
    isPast(date) {
        return new Date(date) < new Date();
    },
    
    /**
     * Check if date is in future
     */
    isFuture(date) {
        return new Date(date) > new Date();
    },
    
    /**
     * Parse date string
     */
    parse(dateString) {
        const d = new Date(dateString);
        return isNaN(d.getTime()) ? null : d;
    },
    
    /**
     * Get current timestamp
     */
    now() {
        return new Date();
    },
    
    /**
     * Get ISO string
     */
    toISO(date = new Date()) {
        return new Date(date).toISOString();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DateUtils;
}

