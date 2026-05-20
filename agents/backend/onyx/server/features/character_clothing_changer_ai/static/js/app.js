/**
 * Main Application Module
 * =======================
 * Initializes and coordinates all modules
 */

// AppState is now managed by StateManager
const AppState = {
    get currentResult() {
        return StateManager ? StateManager.get('currentResult') : null;
    },
    set currentResult(value) {
        if (StateManager) {
            StateManager.set('currentResult', value);
        }
    }
};

/**
 * Initialize application
 */
function initApp() {
    if (typeof AppInitializer !== 'undefined') {
        AppInitializer.init();
    } else {
        // Fallback initialization if AppInitializer is not available
        console.warn('AppInitializer not found, using fallback initialization');
        try {
            Form.init();
            GalleryManager.init();
            HistoryManager.init();
            setupEventHandlers();
            setupEventListeners();
            checkServerStatus();
            setInterval(checkServerStatus, 30000);
        } catch (error) {
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handle(error, { context: 'app initialization' });
            } else {
                console.error('Error initializing app:', error);
            }
        }
    }
}

/**
 * Setup event listeners for EventBus
 */
function setupEventListeners() {
    if (typeof EventBus === 'undefined') return;
    
    // Listen to form events
    EventBus.on('form:submitted', () => {
        if (typeof StateManager !== 'undefined') {
            StateManager.set('isProcessing', true);
        }
    });
    
    EventBus.on('form:completed', (data) => {
        if (typeof StateManager !== 'undefined') {
            StateManager.set('isProcessing', false);
            StateManager.set('currentResult', data);
        }
    });
    
    EventBus.on('form:error', () => {
        if (typeof StateManager !== 'undefined') {
            StateManager.set('isProcessing', false);
        }
    });
    
    // Listen to tab changes
    EventBus.on('tab:changed', (tab) => {
        if (typeof StateManager !== 'undefined') {
            StateManager.set('currentTab', tab);
        }
    });
}

/**
 * Setup event handlers for UI elements
 */
function setupEventHandlers() {
    // Advanced options toggle
    const toggleAdvancedBtn = document.getElementById('toggleAdvancedBtn');
    if (toggleAdvancedBtn) {
        toggleAdvancedBtn.addEventListener('click', () => UI.toggleAdvanced());
    }
    
    // Tab switching
    const tabs = document.querySelectorAll('.tab[data-tab]');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            UI.switchTab(tabName);
        });
    });
    
    // Theme selector
    const themeBtn = document.getElementById('themeBtn');
    if (themeBtn && typeof ThemeManager !== 'undefined') {
        themeBtn.addEventListener('click', () => ThemeManager.toggleMenu());
        ThemeManager.setupClickOutsideHandler();
        ThemeManager.setupThemeOptions();
    } else if (themeBtn) {
        // Fallback
        themeBtn.addEventListener('click', () => UI.toggleThemeMenu());
    }
}

/**
 * Check server status
 */
async function checkServerStatus() {
    try {
        const response = await API.checkHealth();
        
        if (response.success && response.data.status === 'healthy') {
            let text = '';
            let status = 'healthy';
            
            if (response.data.using_deepseek_fallback) {
                text = '🤖 Servidor conectado - Usando DeepSeek (Fallback)';
            } else if (response.data.model_initialized) {
                text = '✅ Servidor conectado - Modelo Flux2 listo';
            } else {
                text = '⏳ Servidor conectado - Modelo cargando...';
                status = 'loading';
            }
            
            if (typeof UI !== 'undefined') {
                UI.updateStatus(status, text);
            }
            
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('server:status:changed', status);
            }
        } else {
            if (typeof UI !== 'undefined') {
                UI.updateStatus('error', '❌ Servidor no disponible');
            }
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('server:status:changed', 'error');
            }
        }
    } catch (error) {
        if (typeof UI !== 'undefined') {
            UI.updateStatus('error', '❌ Error de conexión');
        }
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('server:status:changed', 'error');
        }
        if (typeof Notifications !== 'undefined') {
            Notifications.warning('No se pudo conectar al servidor');
        }
        if (typeof ErrorHandler !== 'undefined') {
            ErrorHandler.handle(error, { context: 'server status check' });
        }
    }
}

/**
 * Global functions for backward compatibility (if needed)
 * These are now handled by event listeners in setupEventHandlers()
 */
function toggleAdvanced() {
    UI.toggleAdvanced();
}

function switchTab(tabName) {
    UI.switchTab(tabName);
}

function toggleThemeMenu() {
    UI.toggleThemeMenu();
}

function setTheme(theme) {
    UI.setTheme(theme);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
