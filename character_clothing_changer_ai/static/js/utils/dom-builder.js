/**
 * DOM Builder Module
 * ===================
 * Utility for building DOM elements
 */

const DOMBuilder = {
    /**
     * Create element with attributes and children
     */
    create(tag, attributes = {}, children = []) {
        const element = document.createElement(tag);
        
        // Set attributes
        for (const [key, value] of Object.entries(attributes)) {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'textContent') {
                element.textContent = value;
            } else if (key === 'innerHTML') {
                element.innerHTML = value;
            } else if (key === 'style' && typeof value === 'object') {
                Object.assign(element.style, value);
            } else if (key === 'onclick') {
                element.onclick = value;
            } else {
                element.setAttribute(key, value);
            }
        }
        
        // Append children
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else if (child instanceof Node) {
                element.appendChild(child);
            }
        });
        
        return element;
    },

    /**
     * Create notification element
     */
    createNotification(message, type = 'info', icon = null) {
        const iconElement = icon || this.getNotificationIcon(type);
        const closeBtn = this.create('button', {
            className: 'notification-close',
            onclick: function() {
                this.closest('.notification').remove();
            }
        }, ['×']);
        
        const content = this.create('div', { className: 'notification-content' }, [
            this.create('span', { className: 'notification-icon' }, [iconElement]),
            this.create('span', { className: 'notification-message' }, [message]),
            closeBtn
        ]);
        
        return this.create('div', {
            className: `notification notification-${type}`
        }, [content]);
    },

    /**
     * Get notification icon
     */
    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    },

    /**
     * Animate element in
     */
    animateIn(element, className = 'show', delay = 10) {
        setTimeout(() => {
            element.classList.add(className);
        }, delay);
    },

    /**
     * Animate element out
     */
    animateOut(element, className = 'show', duration = 300, callback = null) {
        element.classList.remove(className);
        setTimeout(() => {
            if (callback) callback();
            else element.remove();
        }, duration);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DOMBuilder;
}

