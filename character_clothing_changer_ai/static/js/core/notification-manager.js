/**
 * Notification Manager Module
 * ==========================
 * Advanced notification system with queues and priorities
 */

const NotificationManager = {
    /**
     * Notification queue
     */
    queue: [],
    
    /**
     * Active notifications
     */
    active: new Map(),
    
    /**
     * Notification container
     */
    container: null,
    
    /**
     * Max concurrent notifications
     */
    maxConcurrent: 5,
    
    /**
     * Default duration
     */
    defaultDuration: 3000,
    
    /**
     * Initialize notification manager
     */
    init() {
        this.createContainer();
        this.processQueue();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Notification manager initialized');
        }
    },
    
    /**
     * Create notification container
     */
    createContainer() {
        this.container = document.createElement('div');
        this.container.id = 'notification-container';
        this.container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 400px;
        `;
        document.body.appendChild(this.container);
    },
    
    /**
     * Show notification
     */
    show(message, type = 'info', options = {}) {
        const notification = {
            id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            message,
            type,
            duration: options.duration || this.defaultDuration,
            priority: options.priority || 0,
            persistent: options.persistent || false,
            actions: options.actions || [],
            timestamp: Date.now()
        };
        
        // Add to queue
        this.queue.push(notification);
        
        // Sort by priority
        this.queue.sort((a, b) => b.priority - a.priority);
        
        // Process queue
        this.processQueue();
        
        return notification.id;
    },
    
    /**
     * Process notification queue
     */
    processQueue() {
        while (this.active.size < this.maxConcurrent && this.queue.length > 0) {
            const notification = this.queue.shift();
            this.displayNotification(notification);
        }
    },
    
    /**
     * Display notification
     */
    displayNotification(notification) {
        const element = this.createNotificationElement(notification);
        this.container.appendChild(element);
        this.active.set(notification.id, { element, notification });
        
        // Animate in
        requestAnimationFrame(() => {
            element.style.transform = 'translateX(0)';
            element.style.opacity = '1';
        });
        
        // Auto remove if not persistent
        if (!notification.persistent && notification.duration > 0) {
            setTimeout(() => {
                this.remove(notification.id);
            }, notification.duration);
        }
        
        // Emit notification shown event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('notification:shown', notification);
        }
    },
    
    /**
     * Create notification element
     */
    createNotificationElement(notification) {
        const element = document.createElement('div');
        element.className = `notification notification-${notification.type}`;
        element.dataset.notificationId = notification.id;
        element.style.cssText = `
            background: var(--notif-${notification.type}-bg, #fff);
            color: var(--notif-${notification.type}-color, #333);
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateX(400px);
            opacity: 0;
            transition: all 0.3s ease;
            position: relative;
        `;
        
        // Message
        const messageEl = document.createElement('div');
        messageEl.className = 'notification-message';
        messageEl.textContent = notification.message;
        element.appendChild(messageEl);
        
        // Actions
        if (notification.actions.length > 0) {
            const actionsEl = document.createElement('div');
            actionsEl.className = 'notification-actions';
            actionsEl.style.cssText = 'margin-top: 0.5rem; display: flex; gap: 0.5rem;';
            
            notification.actions.forEach(action => {
                const btn = document.createElement('button');
                btn.textContent = action.label;
                btn.className = 'notification-action-btn';
                btn.onclick = () => {
                    if (action.handler) {
                        action.handler();
                    }
                    if (action.dismiss !== false) {
                        this.remove(notification.id);
                    }
                };
                actionsEl.appendChild(btn);
            });
            
            element.appendChild(actionsEl);
        }
        
        // Close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.className = 'notification-close';
        closeBtn.style.cssText = `
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            opacity: 0.5;
        `;
        closeBtn.onclick = () => this.remove(notification.id);
        element.appendChild(closeBtn);
        
        return element;
    },
    
    /**
     * Remove notification
     */
    remove(id) {
        const active = this.active.get(id);
        if (!active) {
            return;
        }
        
        const { element } = active;
        
        // Animate out
        element.style.transform = 'translateX(400px)';
        element.style.opacity = '0';
        
        setTimeout(() => {
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
            this.active.delete(id);
            
            // Process queue
            this.processQueue();
        }, 300);
        
        // Emit notification removed event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('notification:removed', { id });
        }
    },
    
    /**
     * Success notification
     */
    success(message, options = {}) {
        return this.show(message, 'success', options);
    },
    
    /**
     * Error notification
     */
    error(message, options = {}) {
        return this.show(message, 'error', { ...options, duration: options.duration || 5000 });
    },
    
    /**
     * Warning notification
     */
    warning(message, options = {}) {
        return this.show(message, 'warning', { ...options, duration: options.duration || 4000 });
    },
    
    /**
     * Info notification
     */
    info(message, options = {}) {
        return this.show(message, 'info', options);
    },
    
    /**
     * Clear all notifications
     */
    clear() {
        const ids = Array.from(this.active.keys());
        ids.forEach(id => this.remove(id));
        this.queue = [];
    },
    
    /**
     * Get active notifications
     */
    getActive() {
        return Array.from(this.active.values()).map(({ notification }) => notification);
    },
    
    /**
     * Get queue length
     */
    getQueueLength() {
        return this.queue.length;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NotificationManager;
}

