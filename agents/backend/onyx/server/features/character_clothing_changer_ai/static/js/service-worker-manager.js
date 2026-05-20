/**
 * Service Worker Manager Module
 * ==============================
 * Service worker registration and management
 */

const ServiceWorkerManager = {
    /**
     * Service worker registration
     */
    registration: null,
    
    /**
     * Initialize service worker
     */
    async init() {
        if (!('serviceWorker' in navigator)) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Service workers are not supported');
            }
            return false;
        }
        
        try {
            // Register service worker
            this.registration = await navigator.serviceWorker.register('/service-worker.js', {
                scope: '/'
            });
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Service worker registered successfully');
            }
            
            // Listen for updates
            this.registration.addEventListener('updatefound', () => {
                const newWorker = this.registration.installing;
                
                newWorker.addEventListener('statechange', () => {
                    if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                        // New service worker available
                        if (typeof Notifications !== 'undefined') {
                            Notifications.info('Nueva versión disponible. Recarga la página para actualizar.');
                        }
                    }
                });
            });
            
            // Emit service worker ready event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('service-worker:registered', this.registration);
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Service worker registration failed:', error);
            }
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handle(error, { context: 'service worker registration' });
            }
            return false;
        }
    },
    
    /**
     * Unregister service worker
     */
    async unregister() {
        if (!this.registration) {
            return false;
        }
        
        try {
            const unregistered = await this.registration.unregister();
            
            if (unregistered && typeof Logger !== 'undefined') {
                Logger.info('Service worker unregistered');
            }
            
            return unregistered;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Service worker unregistration failed:', error);
            }
            return false;
        }
    },
    
    /**
     * Check for updates
     */
    async checkForUpdates() {
        if (!this.registration) {
            return false;
        }
        
        try {
            await this.registration.update();
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Service worker update check completed');
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Service worker update check failed:', error);
            }
            return false;
        }
    },
    
    /**
     * Get service worker status
     */
    getStatus() {
        if (!('serviceWorker' in navigator)) {
            return 'not_supported';
        }
        
        if (!this.registration) {
            return 'not_registered';
        }
        
        if (navigator.serviceWorker.controller) {
            return 'active';
        }
        
        return 'installing';
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ServiceWorkerManager;
}

