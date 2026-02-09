/**
 * Accordion Manager Module
 * ========================
 * Advanced accordion management with animations and keyboard navigation
 */

const AccordionManager = {
    /**
     * Active accordions
     */
    accordions: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        animation: true,
        keyboardNavigation: true,
        allowMultiple: false,
        collapseOthers: true
    },
    
    /**
     * Initialize accordion manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Accordion manager initialized');
        }
    },
    
    /**
     * Initialize accordion container
     */
    initContainer(container, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const accordionId = `accordion_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const items = container.querySelectorAll('[data-accordion-item]');
        
        items.forEach((item, index) => {
            const header = item.querySelector('[data-accordion-header]');
            const content = item.querySelector('[data-accordion-content]');
            
            if (!header || !content) {
                return;
            }
            
            header.setAttribute('role', 'button');
            header.setAttribute('aria-expanded', 'false');
            header.setAttribute('aria-controls', `accordion-content-${accordionId}-${index}`);
            header.setAttribute('id', `accordion-header-${accordionId}-${index}`);
            header.setAttribute('tabindex', '0');
            
            content.setAttribute('role', 'region');
            content.setAttribute('aria-labelledby', `accordion-header-${accordionId}-${index}`);
            content.setAttribute('id', `accordion-content-${accordionId}-${index}`);
            content.style.display = 'none';
            
            // Setup click handler
            header.addEventListener('click', () => {
                this.toggleItem(item, container, config);
            });
            
            // Setup keyboard navigation
            if (config.keyboardNavigation) {
                this.setupKeyboardNavigation(header, items, container, config);
            }
        });
        
        // Store accordion info
        const accordionInfo = {
            id: accordionId,
            container,
            items: Array.from(items),
            options: config
        };
        
        this.accordions.set(accordionId, accordionInfo);
        
        return accordionId;
    },
    
    /**
     * Toggle accordion item
     */
    toggleItem(item, container, options) {
        const header = item.querySelector('[data-accordion-header]');
        const content = item.querySelector('[data-accordion-content]');
        
        if (!header || !content) {
            return;
        }
        
        const isExpanded = header.getAttribute('aria-expanded') === 'true';
        
        if (isExpanded) {
            this.collapseItem(item, options);
        } else {
            // Collapse others if needed
            if (options.collapseOthers && !options.allowMultiple) {
                const allItems = container.querySelectorAll('[data-accordion-item]');
                allItems.forEach(otherItem => {
                    if (otherItem !== item) {
                        this.collapseItem(otherItem, options);
                    }
                });
            }
            
            this.expandItem(item, options);
        }
    },
    
    /**
     * Expand accordion item
     */
    expandItem(item, options) {
        const header = item.querySelector('[data-accordion-header]');
        const content = item.querySelector('[data-accordion-content]');
        
        if (!header || !content) {
            return;
        }
        
        header.setAttribute('aria-expanded', 'true');
        item.classList.add('expanded');
        
        if (options.animation && typeof AnimationManager !== 'undefined') {
            content.style.display = 'block';
            AnimationManager.fadeIn(content, 300);
        } else {
            content.style.display = 'block';
        }
        
        // Emit expand event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('accordion:expand', { item, header, content });
        }
    },
    
    /**
     * Collapse accordion item
     */
    collapseItem(item, options) {
        const header = item.querySelector('[data-accordion-header]');
        const content = item.querySelector('[data-accordion-content]');
        
        if (!header || !content) {
            return;
        }
        
        header.setAttribute('aria-expanded', 'false');
        item.classList.remove('expanded');
        
        if (options.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeOut(content, 300).then(() => {
                content.style.display = 'none';
            });
        } else {
            content.style.display = 'none';
        }
        
        // Emit collapse event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('accordion:collapse', { item, header, content });
        }
    },
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation(header, allItems, container, options) {
        header.addEventListener('keydown', (e) => {
            const currentItem = header.closest('[data-accordion-item]');
            const currentIndex = Array.from(allItems).indexOf(currentItem);
            
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    const nextIndex = (currentIndex + 1) % allItems.length;
                    const nextHeader = allItems[nextIndex].querySelector('[data-accordion-header]');
                    if (nextHeader) {
                        nextHeader.focus();
                    }
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    const prevIndex = (currentIndex - 1 + allItems.length) % allItems.length;
                    const prevHeader = allItems[prevIndex].querySelector('[data-accordion-header]');
                    if (prevHeader) {
                        prevHeader.focus();
                    }
                    break;
                case 'Home':
                    e.preventDefault();
                    const firstHeader = allItems[0].querySelector('[data-accordion-header]');
                    if (firstHeader) {
                        firstHeader.focus();
                    }
                    break;
                case 'End':
                    e.preventDefault();
                    const lastHeader = allItems[allItems.length - 1].querySelector('[data-accordion-header]');
                    if (lastHeader) {
                        lastHeader.focus();
                    }
                    break;
                case 'Enter':
                case ' ':
                    e.preventDefault();
                    this.toggleItem(currentItem, container, options);
                    break;
            }
        });
    },
    
    /**
     * Expand all items
     */
    expandAll(container, options) {
        const items = container.querySelectorAll('[data-accordion-item]');
        items.forEach(item => {
            this.expandItem(item, options);
        });
    },
    
    /**
     * Collapse all items
     */
    collapseAll(container, options) {
        const items = container.querySelectorAll('[data-accordion-item]');
        items.forEach(item => {
            this.collapseItem(item, options);
        });
    },
    
    /**
     * Get accordion info
     */
    get(accordionId) {
        return this.accordions.get(accordionId);
    },
    
    /**
     * Get all accordions
     */
    getAll() {
        return Array.from(this.accordions.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    AccordionManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AccordionManager;
}

