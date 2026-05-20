/**
 * Tooltip Manager Module
 * ======================
 * Advanced tooltip management with positioning and animations
 */

const TooltipManager = {
    /**
     * Active tooltips
     */
    tooltips: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        position: 'top',
        delay: 200,
        duration: 300,
        animation: true,
        arrow: true,
        offset: 8
    },
    
    /**
     * Initialize tooltip manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Tooltip manager initialized');
        }
    },
    
    /**
     * Show tooltip
     */
    show(element, content, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const tooltipId = `tooltip_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create tooltip element
        const tooltip = this.createTooltipElement(tooltipId, content, config);
        document.body.appendChild(tooltip);
        
        // Position tooltip
        this.positionTooltip(element, tooltip, config);
        
        // Store tooltip info
        const tooltipInfo = {
            id: tooltipId,
            element: tooltip,
            target: element,
            options: config,
            content: content
        };
        
        this.tooltips.set(tooltipId, tooltipInfo);
        
        // Animate in
        if (config.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeIn(tooltip, config.duration);
        } else {
            tooltip.style.display = 'block';
        }
        
        // Emit tooltip show event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('tooltip:show', tooltipInfo);
        }
        
        return tooltipId;
    },
    
    /**
     * Create tooltip element
     */
    createTooltipElement(id, content, options) {
        const tooltip = document.createElement('div');
        tooltip.id = id;
        tooltip.className = `tooltip tooltip-${options.position}`;
        tooltip.setAttribute('role', 'tooltip');
        
        tooltip.innerHTML = `
            ${options.arrow ? '<div class="tooltip-arrow"></div>' : ''}
            <div class="tooltip-content">${content}</div>
        `;
        
        return tooltip;
    },
    
    /**
     * Position tooltip
     */
    positionTooltip(target, tooltip, options) {
        const targetRect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
        const scrollY = window.pageYOffset || document.documentElement.scrollTop;
        
        let top, left;
        
        switch (options.position) {
            case 'top':
                top = targetRect.top + scrollY - tooltipRect.height - options.offset;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (tooltipRect.width / 2);
                break;
            case 'bottom':
                top = targetRect.bottom + scrollY + options.offset;
                left = targetRect.left + scrollX + (targetRect.width / 2) - (tooltipRect.width / 2);
                break;
            case 'left':
                top = targetRect.top + scrollY + (targetRect.height / 2) - (tooltipRect.height / 2);
                left = targetRect.left + scrollX - tooltipRect.width - options.offset;
                break;
            case 'right':
                top = targetRect.top + scrollY + (targetRect.height / 2) - (tooltipRect.height / 2);
                left = targetRect.right + scrollX + options.offset;
                break;
        }
        
        // Keep tooltip within viewport
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        if (left < 0) {
            left = 10;
        } else if (left + tooltipRect.width > viewportWidth) {
            left = viewportWidth - tooltipRect.width - 10;
        }
        
        if (top < 0) {
            top = 10;
        } else if (top + tooltipRect.height > viewportHeight) {
            top = viewportHeight - tooltipRect.height - 10;
        }
        
        tooltip.style.position = 'absolute';
        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
    },
    
    /**
     * Hide tooltip
     */
    hide(tooltipId) {
        const tooltipInfo = this.tooltips.get(tooltipId);
        if (!tooltipInfo) {
            return false;
        }
        
        // Animate out
        if (tooltipInfo.options.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeOut(tooltipInfo.element, tooltipInfo.options.duration).then(() => {
                this.removeTooltip(tooltipInfo);
            });
        } else {
            this.removeTooltip(tooltipInfo);
        }
        
        // Emit tooltip hide event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('tooltip:hide', tooltipInfo);
        }
        
        return true;
    },
    
    /**
     * Remove tooltip
     */
    removeTooltip(tooltipInfo) {
        if (tooltipInfo.element && tooltipInfo.element.parentNode) {
            tooltipInfo.element.parentNode.removeChild(tooltipInfo.element);
        }
        this.tooltips.delete(tooltipInfo.id);
    },
    
    /**
     * Attach tooltip to element
     */
    attach(element, content, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        let tooltipId = null;
        let showTimeout = null;
        
        element.addEventListener('mouseenter', () => {
            showTimeout = setTimeout(() => {
                tooltipId = this.show(element, content, config);
            }, config.delay);
        });
        
        element.addEventListener('mouseleave', () => {
            if (showTimeout) {
                clearTimeout(showTimeout);
            }
            if (tooltipId) {
                this.hide(tooltipId);
                tooltipId = null;
            }
        });
        
        element.addEventListener('focus', () => {
            tooltipId = this.show(element, content, config);
        });
        
        element.addEventListener('blur', () => {
            if (tooltipId) {
                this.hide(tooltipId);
                tooltipId = null;
            }
        });
    },
    
    /**
     * Hide all tooltips
     */
    hideAll() {
        const tooltipIds = Array.from(this.tooltips.keys());
        tooltipIds.forEach(id => this.hide(id));
    },
    
    /**
     * Get tooltip
     */
    get(tooltipId) {
        return this.tooltips.get(tooltipId);
    },
    
    /**
     * Get all tooltips
     */
    getAll() {
        return Array.from(this.tooltips.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    TooltipManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TooltipManager;
}

