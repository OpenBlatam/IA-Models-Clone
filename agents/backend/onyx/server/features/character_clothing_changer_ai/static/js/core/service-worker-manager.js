/**
 * Service Worker Manager Module
 * ============================
 * Manages service worker registration and updates
 */

const ServiceWorkerManager = {
    /**
     * Registered service worker
     */
    registration: null,
    
    /**
     * Service worker script path
     */
    swPath: '/service-worker.js',
    
    /**
     * Check if service workers are supported
     */
    isSupported() {
        return 'serviceWorker' in navigator;
    },
    
    /**
     * Register service worker
     */
    async register() {
        if (!this.isSupported()) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Service workers not supported');
            }
            return false;
        }
        
        try {
            this.registration = await navigator.serviceWorker.register(this.swPath);
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Service worker registered', { scope: this.registration.scope });
            }
            
            // Listen for updates
            this.setupUpdateListener();
            
            // Emit event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('sw:registered', { registration: this.registration });
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Service worker registration failed', error);
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
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Service worker unregistered');
            }
            
            this.registration = null;
            
            // Emit event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('sw:unregistered');
            }
            
            return unregistered;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Service worker unregistration failed', error);
            }
            return false;
        }
    },
    
    /**
     * Setup update listener
     */
    setupUpdateListener() {
        if (!this.registration) return;
        
        // Check for updates periodically
        setInterval(() => {
            this.registration.update();
        }, 60000); // Every minute
        
        // Listen for new service worker
        this.registration.addEventListener('updatefound', () => {
            const newWorker = this.registration.installing;
            
            if (typeof Logger !== 'undefined') {
                Logger.info('New service worker found');
            }
            
            newWorker.addEventListener('statechange', () => {
                if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                    // New service worker available
                    if (typeof Notifications !== 'undefined') {
                        Notifications.info('Nueva versión disponible. Recarga la página para actualizar.');
                    }
                    
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('sw:update_available');
                    }
                } else if (newWorker.state === 'activated') {
                    // Service worker activated
                    if (typeof Logger !== 'undefined') {
                        Logger.info('Service worker activated');
                    }
                    
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('sw:activated');
                    }
                }
            });
        });
    },
    
    /**
     * Update service worker
     */
    async update() {
        if (!this.registration) {
            return false;
        }
        
        try {
            await this.registration.update();
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Service worker update checked');
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Service worker update failed', error);
            }
            return false;
        }
    },
    
    /**
     * Skip waiting (activate new service worker immediately)
     */
    async skipWaiting() {
        if (!this.registration || !this.registration.waiting) {
            return false;
        }
        
        try {
            this.registration.waiting.postMessage({ type: 'SKIP_WAITING' });
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Skip waiting message sent');
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Skip waiting failed', error);
            }
            return false;
        }
    },
    
    /**
     * Get service worker status
     */
    getStatus() {
        if (!this.isSupported()) {
            return { supported: false };
        }
        
        return {
            supported: true,
            registered: this.registration !== null,
            scope: this.registration?.scope,
            state: this.registration?.active?.state || 'none'
        };
    },
    
    /**
     * Initialize service worker manager
     */
    init() {
        if (!this.isSupported()) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Service workers not supported in this browser');
            }
            return;
        }
        
        // Register service worker
        this.register();
        
        // Listen for controller change
        navigator.serviceWorker.addEventListener('controllerchange', () => {
            if (typeof Logger !== 'undefined') {
                Logger.info('Service worker controller changed');
            }
            
            // Reload page if needed
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('sw:controller_changed');
            }
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Service worker manager initialized');
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => ServiceWorkerManager.init());
    } else {
        ServiceWorkerManager.init();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ServiceWorkerManager;
}

