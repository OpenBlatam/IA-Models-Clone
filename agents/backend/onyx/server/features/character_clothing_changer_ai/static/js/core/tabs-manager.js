/**
 * Tabs Manager Module
 * ===================
 * Advanced tabs management with keyboard navigation and animations
 */

const TabsManager = {
    /**
     * Active tabs
     */
    tabs: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        animation: true,
        keyboardNavigation: true,
        rememberActive: false,
        storageKey: null
    },
    
    /**
     * Initialize tabs manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Tabs manager initialized');
        }
    },
    
    /**
     * Initialize tabs container
     */
    initContainer(container, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const tabsId = `tabs_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const tabButtons = container.querySelectorAll('[data-tab]');
        const tabPanels = container.querySelectorAll('[data-tab-panel]');
        
        // Setup tab buttons
        tabButtons.forEach((button, index) => {
            const tabId = button.getAttribute('data-tab');
            const panel = container.querySelector(`[data-tab-panel="${tabId}"]`);
            
            if (!panel) {
                return;
            }
            
            button.setAttribute('role', 'tab');
            button.setAttribute('aria-controls', `tab-panel-${tabId}`);
            button.setAttribute('id', `tab-${tabId}`);
            
            panel.setAttribute('role', 'tabpanel');
            panel.setAttribute('aria-labelledby', `tab-${tabId}`);
            panel.setAttribute('id', `tab-panel-${tabId}`);
            
            // Setup click handler
            button.addEventListener('click', () => {
                this.activateTab(container, tabId, config);
            });
            
            // Setup keyboard navigation
            if (config.keyboardNavigation) {
                this.setupKeyboardNavigation(button, tabButtons, container, config);
            }
        });
        
        // Store tabs info
        const tabsInfo = {
            id: tabsId,
            container,
            buttons: Array.from(tabButtons),
            panels: Array.from(tabPanels),
            options: config
        };
        
        this.tabs.set(tabsId, tabsInfo);
        
        // Load saved active tab
        if (config.rememberActive && config.storageKey) {
            const savedTab = localStorage.getItem(config.storageKey);
            if (savedTab) {
                this.activateTab(container, savedTab, config);
            } else {
                // Activate first tab
                const firstTab = tabButtons[0]?.getAttribute('data-tab');
                if (firstTab) {
                    this.activateTab(container, firstTab, config);
                }
            }
        } else {
            // Activate first tab
            const firstTab = tabButtons[0]?.getAttribute('data-tab');
            if (firstTab) {
                this.activateTab(container, firstTab, config);
            }
        }
        
        return tabsId;
    },
    
    /**
     * Activate tab
     */
    activateTab(container, tabId, options) {
        const tabButtons = container.querySelectorAll('[data-tab]');
        const tabPanels = container.querySelectorAll('[data-tab-panel]');
        
        // Deactivate all tabs
        tabButtons.forEach(button => {
            const btnTabId = button.getAttribute('data-tab');
            const panel = container.querySelector(`[data-tab-panel="${btnTabId}"]`);
            
            button.classList.remove('active');
            button.setAttribute('aria-selected', 'false');
            button.setAttribute('tabindex', '-1');
            
            if (panel) {
                panel.classList.remove('active');
                panel.style.display = 'none';
            }
        });
        
        // Activate selected tab
        const activeButton = container.querySelector(`[data-tab="${tabId}"]`);
        const activePanel = container.querySelector(`[data-tab-panel="${tabId}"]`);
        
        if (activeButton && activePanel) {
            activeButton.classList.add('active');
            activeButton.setAttribute('aria-selected', 'true');
            activeButton.setAttribute('tabindex', '0');
            activeButton.focus();
            
            if (options.animation && typeof AnimationManager !== 'undefined') {
                activePanel.style.display = 'block';
                AnimationManager.fadeIn(activePanel, 300);
            } else {
                activePanel.style.display = 'block';
            }
            
            activePanel.classList.add('active');
            
            // Save active tab
            if (options.rememberActive && options.storageKey) {
                localStorage.setItem(options.storageKey, tabId);
            }
            
            // Emit tab change event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('tabs:change', { tabId, container });
            }
        }
    },
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation(button, allButtons, container, options) {
        button.addEventListener('keydown', (e) => {
            const currentIndex = Array.from(allButtons).indexOf(button);
            let newIndex = currentIndex;
            
            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    newIndex = (currentIndex - 1 + allButtons.length) % allButtons.length;
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    newIndex = (currentIndex + 1) % allButtons.length;
                    break;
                case 'Home':
                    e.preventDefault();
                    newIndex = 0;
                    break;
                case 'End':
                    e.preventDefault();
                    newIndex = allButtons.length - 1;
                    break;
            }
            
            if (newIndex !== currentIndex) {
                const newButton = allButtons[newIndex];
                const tabId = newButton.getAttribute('data-tab');
                this.activateTab(container, tabId, options);
            }
        });
    },
    
    /**
     * Get tabs info
     */
    get(tabsId) {
        return this.tabs.get(tabsId);
    },
    
    /**
     * Get all tabs
     */
    getAll() {
        return Array.from(this.tabs.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    TabsManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TabsManager;
}

