/**
 * Popover Manager Module
 * ======================
 * Advanced popover management with positioning and interactions
 */

const PopoverManager = {
    /**
     * Active popovers
     */
    popovers: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        position: 'bottom',
        trigger: 'click',
        closable: true,
        closeOnOutsideClick: true,
        animation: true,
        arrow: true,
        offset: 8
    },
    
    /**
     * Initialize popover manager
     */
    init() {
        // Setup outside click handler
        document.addEventListener('click', (e) => {
            this.popovers.forEach((popoverInfo) => {
                if (popoverInfo.options.closeOnOutsideClick) {
                    const isClickInside = popoverInfo.element.contains(e.target) || 
                                        popoverInfo.target.contains(e.target);
                    if (!isClickInside) {
                        this.hide(popoverInfo.id);
                    }
                }
            });
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Popover manager initialized');
        }
    },
    
    /**
     * Show popover
     */
    show(element, content, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const popoverId = `popover_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create popover element
        const popover = this.createPopoverElement(popoverId, content, config);
        document.body.appendChild(popover);
        
        // Position popover
        this.positionPopover(element, popover, config);
        
        // Store popover info
        const popoverInfo = {
            id: popoverId,
            element: popover,
            target: element,
            options: config,
            content: content
        };
        
        this.popovers.set(popoverId, popoverInfo);
        
        // Animate in
        if (config.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeIn(popover, 300);
        } else {
            popover.style.display = 'block';
        }
        
        // Emit popover show event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('popover:show', popoverInfo);
        }
        
        return popoverId;
    },
    
    /**
     * Create popover element
     */
    createPopoverElement(id, content, options) {
        const popover = document.createElement('div');
        popover.id = id;
        popover.className = `popover popover-${options.position}`;
        popover.setAttribute('role', 'tooltip');
        
        popover.innerHTML = `
            ${options.arrow ? '<div class="popover-arrow"></div>' : ''}
            ${options.closable ? `
                <button class="popover-close" aria-label="Cerrar" data-popover-close="${id}">
                    <span>&times;</span>
                </button>
            ` : ''}
            <div class="popover-content">${content}</div>
        `;
        
        // Setup close button
        if (options.closable) {
            const closeBtn = popover.querySelector(`[data-popover-close="${id}"]`);
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    this.hide(id);
                });
            }
        }
        
        return popover;
    },
    
    /**
     * Position popover
     */
    positionPopover(target, popover, options) {
        const targetRect = target.getBoundingClientRect();
        const popoverRect = popover.getBoundingClientRect();
        const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
        const scrollY = window.pageYOffset || document.documentElement.scrollTop;
        
        let top, left;
        
        switch (options.position) {
            case 'top':
                top = targetRect.top + scrollY - popoverRect.height - options.offset;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (popoverRect.width / 2);
                break;
            case 'bottom':
                top = targetRect.bottom + scrollY + options.offset;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (popoverRect.width / 2);
                break;
            case 'left':
                top = targetRect.top + scrollY + (targetRect.height / 2) - (popoverRect.height / 2);
                left = targetRect.left + scrollX - popoverRect.width - options.offset;
                break;
            case 'right':
                top = targetRect.top + scrollY + (targetRect.height / 2) - (popoverRect.height / 2);
                left = targetRect.right + scrollX + options.offset;
                break;
        }
        
        // Keep popover within viewport
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        if (left < 0) {
            left = 10;
        } else if (left + popoverRect.width > viewportWidth) {
            left = viewportWidth - popoverRect.width - 10;
        }
        
        if (top < 0) {
            top = 10;
        } else if (top + popoverRect.height > viewportHeight) {
            top = viewportHeight - popoverRect.height - 10;
        }
        
        popover.style.position = 'absolute';
        popover.style.top = `${top}px`;
        popover.style.left = `${left}px`;
    },
    
    /**
     * Hide popover
     */
    hide(popoverId) {
        const popoverInfo = this.popovers.get(popoverId);
        if (!popoverInfo) {
            return false;
        }
        
        // Animate out
        if (popoverInfo.options.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeOut(popoverInfo.element, 300).then(() => {
                this.removePopover(popoverInfo);
            });
        } else {
            this.removePopover(popoverInfo);
        }
        
        // Emit popover hide event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('popover:hide', popoverInfo);
        }
        
        return true;
    },
    
    /**
     * Remove popover
     */
    removePopover(popoverInfo) {
        if (popoverInfo.element && popoverInfo.element.parentNode) {
            popoverInfo.element.parentNode.removeChild(popoverInfo.element);
        }
        this.popovers.delete(popoverInfo.id);
    },
    
    /**
     * Attach popover to element
     */
    attach(element, content, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        let popoverId = null;
        
        if (config.trigger === 'click') {
            element.addEventListener('click', (e) => {
                e.stopPropagation();
                if (popoverId) {
                    this.hide(popoverId);
                    popoverId = null;
                } else {
                    popoverId = this.show(element, content, config);
                }
            });
        } else if (config.trigger === 'hover') {
            element.addEventListener('mouseenter', () => {
                popoverId = this.show(element, content, config);
            });
            
            element.addEventListener('mouseleave', () => {
                if (popoverId) {
                    this.hide(popoverId);
                    popoverId = null;
                }
            });
        }
    },
    
    /**
     * Hide all popovers
     */
    hideAll() {
        const popoverIds = Array.from(this.popovers.keys());
        popoverIds.forEach(id => this.hide(id));
    },
    
    /**
     * Get popover
     */
    get(popoverId) {
        return this.popovers.get(popoverId);
    },
    
    /**
     * Get all popovers
     */
    getAll() {
        return Array.from(this.popovers.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    PopoverManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PopoverManager;
}

