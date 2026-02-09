/**
 * Application Bootstrap
 * =====================
 * Handles module loading, error handling, and application initialization
 */

/**
 * Module configuration for dynamic loading
 * Organized by category for better dependency management
 */
const MODULE_CONFIG = {
    core: [
        'config', 'storage', 'logger', 'event-bus',
        'state-manager', 'error-handler', 'hot-reload', 'plugin-system'
    ],
    utils: [
        'local-storage-wrapper', 'dom-builder', 'cache', 'api-cache',
        'api-performance', 'file-handler', 'drag-drop-handler',
        'form-submission-handler', 'item-manager-base', 'date-utils',
        'format-utils', 'object-utils', 'debounce', 'api', 'utils',
        'form-data-builder', 'validator', 'constants', 'performance-optimizer', 'helpers'
    ],
    ui: [
        'ui', 'image-analyzer', 'progress', 'comparison',
        'gallery', 'history', 'form'
    ],
    features: [
        'notifications', 'favorites', 'filters', 'shortcuts',
        'stats', 'advanced-analytics'
    ],
    renderers: [
        'item-renderer', 'modal-viewer', 'image-stats-calculator',
        'file-downloader', 'config-exporter', 'search-filter', 'stats-calculator'
    ],
    system: [
        'module-loader', 'health-monitor', 'plugin-system', 'middleware',
        'config-manager', 'debug-tools', 'security-manager', 'offline-manager',
        'service-worker-manager', 'i18n', 'version-manager', 'sync-manager',
        'backup-manager', 'component-registry', 'metrics-collector',
        'feature-flags', 'documentation-generator', 'test-runner',
        'resource-manager', 'queue-manager', 'image-processor',
        'theme-manager', 'bundle-optimizer', 'accessibility-manager'
    ]
};

/**
 * Initialize module loading system
 * Falls back to direct loading if ModuleLoaderV2 is unavailable
 */
function initializeModules() {
    if (typeof ModuleLoaderV2 === 'undefined') {
        console.warn('ModuleLoaderV2 not available, loading modules directly');
        loadModulesDirectly();
    } else {
        loadModulesWithLoader();
    }
}

/**
 * Fallback: Load all modules directly
 */
function loadModulesDirectly() {
    Object.keys(MODULE_CONFIG).forEach(category => {
        MODULE_CONFIG[category].forEach(moduleName => {
            const script = document.createElement('script');
            script.src = `static/js/${category}/${moduleName}.js`;
            script.async = true;
            script.onerror = () => {
                console.warn(`Failed to load module: ${category}/${moduleName}`);
            };
            document.head.appendChild(script);
        });
    });
}

/**
 * Load modules using ModuleLoaderV2
 */
function loadModulesWithLoader() {
    ModuleLoaderV2.init();

    Object.keys(MODULE_CONFIG).forEach(category => {
        MODULE_CONFIG[category].forEach(moduleName => {
            ModuleLoaderV2.loadModule(moduleName, category).catch(err => {
                console.warn(`Failed to load module ${category}/${moduleName}:`, err);
            });
        });
    });
}

/**
 * Global Error Handler
 * Catches unhandled errors and displays them to the user
 */
function setupErrorHandlers() {
    window.addEventListener('error', function (event) {
        console.error('Global error:', event.error);
        const errorBoundary = document.getElementById('errorBoundary');
        const errorMessage = document.getElementById('errorMessage');

        if (errorBoundary && errorMessage) {
            errorMessage.textContent = event.error?.message || 'Ha ocurrido un error inesperado';
            errorBoundary.classList.remove('hidden');
        }
    });

    /**
     * Unhandled Promise Rejection Handler
     */
    window.addEventListener('unhandledrejection', function (event) {
        console.error('Unhandled promise rejection:', event.reason);
        const errorBoundary = document.getElementById('errorBoundary');
        const errorMessage = document.getElementById('errorMessage');

        if (errorBoundary && errorMessage) {
            errorMessage.textContent = event.reason?.message || 'Error al procesar la solicitud';
            errorBoundary.classList.remove('hidden');
        }
    });
}

/**
 * Setup reload button handler
 */
function setupReloadButton() {
    const reloadBtn = document.getElementById('reloadBtn');
    if (reloadBtn) {
        reloadBtn.addEventListener('click', function () {
            window.location.reload();
        });
    }
}

/**
 * Hide loading overlay when DOM is ready
 */
function hideLoadingOverlay() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        setTimeout(() => {
            loadingOverlay.classList.add('hidden');
        }, 500);
    }
}

/**
 * Initialize application when DOM is ready
 */
function initializeApp() {
    setupErrorHandlers();
    setupReloadButton();
    hideLoadingOverlay();
    initializeModules();
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.AppBootstrap = {
        MODULE_CONFIG,
        initializeModules,
        setupErrorHandlers
    };
}

