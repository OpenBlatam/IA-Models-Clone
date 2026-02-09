/**
 * App Initializer Module
 * ========================
 * Handles application initialization and module coordination
 */

const AppInitializer = {
    /**
     * Initialize all application modules
     */
    init() {
        try {
            // Initialize infrastructure modules first
            this.initInfrastructure();
            
            // Initialize core modules
            this.initCore();
            
            // Initialize feature modules
            this.initFeatures();
            
            // Initialize UI modules
            this.initUI();
            
            // Setup event handlers and listeners
            this.setupHandlers();
            
            // Setup periodic tasks
            this.setupPeriodicTasks();
            
            // Apply saved settings
            this.applySavedSettings();
            
            // Show welcome notification
            this.showWelcome();
            
            // Emit initialization complete event
            this.emitInitialized();
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Application initialized successfully');
            }
        } catch (error) {
            this.handleInitError(error);
        }
    },

    /**
     * Initialize infrastructure modules
     */
    initInfrastructure() {
        if (typeof ModuleLoader !== 'undefined') {
            ModuleLoader.init();
        }
        
        if (typeof ConfigManager !== 'undefined') {
            ConfigManager.init();
        }
        
        if (typeof DebugTools !== 'undefined' && DebugTools.enabled) {
            DebugTools.init();
        }
        
        if (typeof PerformanceMonitor !== 'undefined') {
            PerformanceMonitor.init();
        }
        
        if (typeof Analytics !== 'undefined') {
            Analytics.init();
        }
        
        if (typeof HealthMonitor !== 'undefined') {
            HealthMonitor.init();
        }
        
        if (typeof PluginSystem !== 'undefined' && typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:system:ready');
        }
        
        if (typeof OfflineManager !== 'undefined') {
            OfflineManager.init();
        }
        
        if (typeof ServiceWorkerManager !== 'undefined') {
            ServiceWorkerManager.init();
        }
        
        if (typeof I18n !== 'undefined') {
            I18n.init();
        }
        
        if (typeof VersionManager !== 'undefined') {
            VersionManager.init();
        }
        
        if (typeof SyncManager !== 'undefined') {
            SyncManager.init();
        }
        
        if (typeof BackupManager !== 'undefined') {
            BackupManager.init();
        }
        
        if (typeof ComponentRegistry !== 'undefined') {
            ComponentRegistry.init();
        }
        
        if (typeof DevTools !== 'undefined') {
            DevTools.init();
        }
        
        if (typeof FeatureFlags !== 'undefined') {
            FeatureFlags.init();
        }
        
        // Initialize performance monitor
        if (typeof PerformanceMonitor !== 'undefined') {
            PerformanceMonitor.init();
        }
        
        // Initialize resource manager
        if (typeof ResourceManager !== 'undefined') {
            ResourceManager.init();
        }
        
        // Initialize validation engine
        if (typeof ValidationEngine !== 'undefined') {
            ValidationEngine.init();
        }
        
        // Initialize cache manager V2
        if (typeof CacheManagerV2 !== 'undefined') {
            CacheManagerV2.init({
                strategy: 'lru',
                maxSize: 100,
                defaultTTL: 3600000
            });
        }
        
        // Initialize notification manager
        if (typeof NotificationManager !== 'undefined') {
            NotificationManager.init();
        }
        
        if (typeof MetricsCollector !== 'undefined' && 
            typeof FeatureFlags !== 'undefined' && 
            FeatureFlags.isEnabled('enableAnalytics')) {
            MetricsCollector.init();
        }
        
        if (typeof ThemeManager !== 'undefined') {
            ThemeManager.init();
        }
        
        // Initialize theme engine (alternative to ThemeManager)
        if (typeof ThemeEngine !== 'undefined') {
            ThemeEngine.init();
        }
        
        if (typeof AccessibilityManager !== 'undefined') {
            AccessibilityManager.init();
        }
        
        // Initialize animation manager
        if (typeof AnimationManager !== 'undefined') {
            AnimationManager.init();
        }
        
        // Translation manager is auto-initialized
        
        // Initialize image processor V2
        if (typeof ImageProcessorV2 !== 'undefined') {
            ImageProcessorV2.init();
        }
        
        // Form validator V2 is auto-initialized
        
        // Storage manager V2 is auto-initialized
        
        // Initialize API client V2
        if (typeof ApiClientV2 !== 'undefined') {
            ApiClientV2.init({
                baseURL: window.location.origin,
                retry: {
                    maxRetries: 3,
                    retryDelay: 1000
                }
            });
        }
        
        // Drag drop manager is auto-initialized
        
        // Upload manager is auto-initialized
        
        // Download manager is auto-initialized
        
        // Clipboard manager is auto-initialized
        
        // Modal manager is auto-initialized
        
        // Tooltip manager is auto-initialized
        
        // Popover manager is auto-initialized
        
        // Dropdown manager is auto-initialized
        
        // Tabs manager is auto-initialized
        
        // Accordion manager is auto-initialized
        
        // Carousel manager is auto-initialized
        
        // Collapse manager is auto-initialized
        
        // Initialize router
        if (typeof Router !== 'undefined') {
            Router.init();
        }
        
        // Initialize command pattern
        if (typeof CommandPattern !== 'undefined') {
            CommandPattern.init();
        }
        
        // Event store is auto-initialized
        
        // Initialize worker manager
        if (typeof WorkerManager !== 'undefined') {
            WorkerManager.init();
        }
        
        // Initialize stream manager
        if (typeof StreamManager !== 'undefined') {
            StreamManager.init();
        }
        
        // Initialize rate limiter
        if (typeof RateLimiter !== 'undefined') {
            // Create default rate limiters
            RateLimiter.create('api', { maxRequests: 10, windowMs: 60000 });
            RateLimiter.create('image-processing', { maxRequests: 5, windowMs: 60000 });
        }
        
        // ResourceManager, QueueManager, ImageProcessor don't need init
        // They are ready to use immediately
    },

    /**
     * Initialize core modules
     */
    initCore() {
        if (typeof StateManager !== 'undefined' && typeof Storage !== 'undefined') {
            const savedTheme = Storage.getTheme();
            if (savedTheme) {
                StateManager.set('theme', savedTheme);
            }
        }
        
        if (typeof Favorites !== 'undefined') {
            Favorites.init();
        }
    },

    /**
     * Initialize feature modules
     */
    initFeatures() {
        if (typeof Shortcuts !== 'undefined') {
            Shortcuts.init();
        }
    },

    /**
     * Initialize UI modules
     */
    initUI() {
        if (typeof Form !== 'undefined') {
            Form.init();
        }
        if (typeof GalleryManager !== 'undefined') {
            GalleryManager.init();
        }
        if (typeof HistoryManager !== 'undefined') {
            HistoryManager.init();
        }
    },

    /**
     * Setup event handlers and listeners
     */
    setupHandlers() {
        // This will be called from app.js
        if (typeof setupEventHandlers === 'function') {
            setupEventHandlers();
        }
        if (typeof setupEventListeners === 'function') {
            setupEventListeners();
        }
    },

    /**
     * Setup periodic tasks
     */
    setupPeriodicTasks() {
        if (typeof checkServerStatus === 'function') {
            checkServerStatus();
            setInterval(checkServerStatus, 30000);
        }
    },

    /**
     * Apply saved settings
     */
    applySavedSettings() {
        // Theme is now handled by ThemeManager
        if (typeof ThemeManager !== 'undefined') {
            // ThemeManager already applies saved theme in init()
            return;
        }
        
        // Fallback to old method
        if (typeof StateManager !== 'undefined' && typeof UI !== 'undefined') {
            const theme = StateManager.get('theme');
            if (theme) {
                UI.setTheme(theme);
            }
        } else if (typeof Storage !== 'undefined' && typeof UI !== 'undefined') {
            const savedTheme = Storage.getTheme();
            if (savedTheme) {
                UI.setTheme(savedTheme);
            }
        }
    },

    /**
     * Show welcome notification
     */
    showWelcome() {
        if (typeof Notifications !== 'undefined') {
            Notifications.info('Aplicación inicializada correctamente', 2000);
        }
    },

    /**
     * Emit initialization complete event
     */
    emitInitialized() {
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('app:initialized');
        }
    },

    /**
     * Handle initialization errors
     */
    handleInitError(error) {
        if (typeof ErrorHandler !== 'undefined') {
            ErrorHandler.handle(error, { context: 'app initialization' });
        } else {
            console.error('Error initializing app:', error);
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AppInitializer;
}

