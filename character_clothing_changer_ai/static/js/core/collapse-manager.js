/**
 * Collapse Manager Module
 * ======================
 * Advanced collapse/expand functionality with animations
 */

const CollapseManager = {
    /**
     * Active collapses
     */
    collapses: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        animation: true,
        animationDuration: 300,
        toggle: true
    },
    
    /**
     * Initialize collapse manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Collapse manager initialized');
        }
    },
    
    /**
     * Initialize collapse element
     */
    initElement(element, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const collapseId = `collapse_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const trigger = element.querySelector('[data-collapse-trigger]') || element;
        const content = element.querySelector('[data-collapse-content]');
        
        if (!content) {
            return null;
        }
        
        let isExpanded = false;
        
        // Setup initial state
        content.style.display = 'none';
        content.style.overflow = 'hidden';
        content.style.transition = config.animation ? `height ${config.animationDuration}ms ease` : 'none';
        
        // Expand function
        const expand = () => {
            if (isExpanded) {
                return;
            }
            
            content.style.display = 'block';
            const height = content.scrollHeight;
            content.style.height = '0px';
            
            // Force reflow
            content.offsetHeight;
            
            content.style.height = `${height}px`;
            
            isExpanded = true;
            element.classList.add('expanded');
            
            // Emit expand event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('collapse:expand', { collapseId, element, content });
            }
            
            // Cleanup after animation
            if (config.animation) {
                setTimeout(() => {
                    content.style.height = 'auto';
                }, config.animationDuration);
            }
        };
        
        // Collapse function
        const collapse = () => {
            if (!isExpanded) {
                return;
            }
            
            const height = content.scrollHeight;
            content.style.height = `${height}px`;
            
            // Force reflow
            content.offsetHeight;
            
            content.style.height = '0px';
            
            isExpanded = false;
            element.classList.remove('expanded');
            
            // Emit collapse event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('collapse:collapse', { collapseId, element, content });
            }
            
            // Hide after animation
            if (config.animation) {
                setTimeout(() => {
                    content.style.display = 'none';
                }, config.animationDuration);
            } else {
                content.style.display = 'none';
            }
        };
        
        // Toggle function
        const toggle = () => {
            if (isExpanded) {
                collapse();
            } else {
                expand();
            }
        };
        
        // Setup trigger
        trigger.addEventListener('click', () => {
            if (config.toggle) {
                toggle();
            } else {
                if (isExpanded) {
                    collapse();
                } else {
                    expand();
                }
            }
        });
        
        // Store collapse info
        const collapseInfo = {
            id: collapseId,
            element,
            trigger,
            content,
            isExpanded: false,
            options: config,
            expand,
            collapse,
            toggle
        };
        
        this.collapses.set(collapseId, collapseInfo);
        
        return collapseId;
    },
    
    /**
     * Expand collapse
     */
    expand(collapseId) {
        const collapseInfo = this.collapses.get(collapseId);
        if (collapseInfo) {
            collapseInfo.expand();
        }
    },
    
    /**
     * Collapse element
     */
    collapse(collapseId) {
        const collapseInfo = this.collapses.get(collapseId);
        if (collapseInfo) {
            collapseInfo.collapse();
        }
    },
    
    /**
     * Toggle collapse
     */
    toggle(collapseId) {
        const collapseInfo = this.collapses.get(collapseId);
        if (collapseInfo) {
            collapseInfo.toggle();
        }
    },
    
    /**
     * Get collapse info
     */
    get(collapseId) {
        return this.collapses.get(collapseId);
    },
    
    /**
     * Get all collapses
     */
    getAll() {
        return Array.from(this.collapses.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    CollapseManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CollapseManager;
}

