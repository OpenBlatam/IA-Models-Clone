/**
 * Notifications Module
 * ===================
 * Handles toast notifications
 */

const Notifications = {
    /**
     * Show notification
     */
    show(message, type = 'info', duration = 3000) {
        // Sanitize message
        if (typeof SecurityManager !== 'undefined') {
            message = SecurityManager.escapeHTML(message);
        }
        const notification = DOMBuilder.createNotification(message, type);
        document.body.appendChild(notification);

        // Animate in
        DOMBuilder.animateIn(notification);

        // Auto remove
        if (duration > 0) {
            setTimeout(() => {
                DOMBuilder.animateOut(notification);
            }, duration);
        }
        
        return notification;
    },

    /**
     * Success notification
     */
    success(message, duration = 3000) {
        this.show(message, 'success', duration);
    },

    /**
     * Error notification
     */
    error(message, duration = 5000) {
        this.show(message, 'error', duration);
    },

    /**
     * Warning notification
     */
    warning(message, duration = 4000) {
        this.show(message, 'warning', duration);
    },

    /**
     * Info notification
     */
    info(message, duration = 3000) {
        this.show(message, 'info', duration);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Notifications;
}


