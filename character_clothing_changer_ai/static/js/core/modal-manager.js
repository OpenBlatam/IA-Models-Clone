/**
 * Modal Manager Module
 * ====================
 * Advanced modal management with animations and stacking
 */

const ModalManager = {
    /**
     * Active modals
     */
    modals: new Map(),
    
    /**
     * Modal stack
     */
    stack: [],
    
    /**
     * Default options
     */
    defaultOptions: {
        closable: true,
        closeOnBackdrop: true,
        closeOnEscape: true,
        animation: true,
        backdrop: true,
        zIndex: 1000
    },
    
    /**
     * Initialize modal manager
     */
    init() {
        // Setup escape key handler
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.stack.length > 0) {
                const topModal = this.stack[this.stack.length - 1];
                if (topModal.options.closeOnEscape) {
                    this.close(topModal.id);
                }
            }
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Modal manager initialized');
        }
    },
    
    /**
     * Create modal
     */
    create(content, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const modalId = `modal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create modal element
        const modal = this.createModalElement(modalId, content, config);
        
        // Create backdrop
        if (config.backdrop) {
            const backdrop = this.createBackdrop(modalId, config);
            document.body.appendChild(backdrop);
        }
        
        document.body.appendChild(modal);
        
        // Store modal info
        const modalInfo = {
            id: modalId,
            element: modal,
            backdrop: config.backdrop ? document.querySelector(`[data-modal-backdrop="${modalId}"]`) : null,
            options: config,
            content: content
        };
        
        this.modals.set(modalId, modalInfo);
        this.stack.push(modalInfo);
        
        // Update z-index
        this.updateZIndex();
        
        // Animate in
        if (config.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeIn(modal, 300);
            if (modalInfo.backdrop) {
                AnimationManager.fadeIn(modalInfo.backdrop, 300);
            }
        } else {
            modal.style.display = 'block';
        }
        
        // Focus management
        this.trapFocus(modal);
        
        // Emit modal open event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('modal:open', modalInfo);
        }
        
        return modalId;
    },
    
    /**
     * Create modal element
     */
    createModalElement(id, content, options) {
        const modal = document.createElement('div');
        modal.id = id;
        modal.className = 'modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-labelledby', `${id}-title`);
        
        modal.innerHTML = `
            <div class="modal-content">
                ${options.closable ? `
                    <button class="modal-close" aria-label="Cerrar" data-modal-close="${id}">
                        <span>&times;</span>
                    </button>
                ` : ''}
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;
        
        // Setup close button
        if (options.closable) {
            const closeBtn = modal.querySelector(`[data-modal-close="${id}"]`);
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    this.close(id);
                });
            }
        }
        
        // Setup backdrop click
        if (options.closeOnBackdrop) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.close(id);
                }
            });
        }
        
        return modal;
    },
    
    /**
     * Create backdrop
     */
    createBackdrop(modalId, options) {
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop';
        backdrop.setAttribute('data-modal-backdrop', modalId);
        
        if (options.closeOnBackdrop) {
            backdrop.addEventListener('click', () => {
                this.close(modalId);
            });
        }
        
        return backdrop;
    },
    
    /**
     * Close modal
     */
    close(modalId) {
        const modalInfo = this.modals.get(modalId);
        if (!modalInfo) {
            return false;
        }
        
        // Animate out
        if (modalInfo.options.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeOut(modalInfo.element, 300).then(() => {
                this.removeModal(modalInfo);
            });
            if (modalInfo.backdrop) {
                AnimationManager.fadeOut(modalInfo.backdrop, 300);
            }
        } else {
            this.removeModal(modalInfo);
        }
        
        // Emit modal close event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('modal:close', modalInfo);
        }
        
        return true;
    },
    
    /**
     * Remove modal
     */
    removeModal(modalInfo) {
        if (modalInfo.element && modalInfo.element.parentNode) {
            modalInfo.element.parentNode.removeChild(modalInfo.element);
        }
        
        if (modalInfo.backdrop && modalInfo.backdrop.parentNode) {
            modalInfo.backdrop.parentNode.removeChild(modalInfo.backdrop);
        }
        
        // Remove from stack
        const index = this.stack.findIndex(m => m.id === modalInfo.id);
        if (index > -1) {
            this.stack.splice(index, 1);
        }
        
        this.modals.delete(modalInfo.id);
        
        // Update z-index
        this.updateZIndex();
        
        // Restore focus
        this.restoreFocus();
    },
    
    /**
     * Update z-index
     */
    updateZIndex() {
        this.stack.forEach((modal, index) => {
            const zIndex = this.defaultOptions.zIndex + index;
            modal.element.style.zIndex = zIndex;
            if (modal.backdrop) {
                modal.backdrop.style.zIndex = zIndex - 1;
            }
        });
    },
    
    /**
     * Trap focus
     */
    trapFocus(modal) {
        const focusableElements = modal.querySelectorAll(
            'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        
        if (focusableElements.length > 0) {
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            modal.addEventListener('keydown', (e) => {
                if (e.key === 'Tab') {
                    if (e.shiftKey) {
                        if (document.activeElement === firstElement) {
                            e.preventDefault();
                            lastElement.focus();
                        }
                    } else {
                        if (document.activeElement === lastElement) {
                            e.preventDefault();
                            firstElement.focus();
                        }
                    }
                }
            });
            
            firstElement.focus();
        }
    },
    
    /**
     * Restore focus
     */
    restoreFocus() {
        // Focus management would go here
        // Store previous focus element before opening modal
    },
    
    /**
     * Close all modals
     */
    closeAll() {
        const modalIds = Array.from(this.modals.keys());
        modalIds.forEach(id => this.close(id));
    },
    
    /**
     * Get modal
     */
    get(modalId) {
        return this.modals.get(modalId);
    },
    
    /**
     * Get all modals
     */
    getAll() {
        return Array.from(this.modals.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    ModalManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModalManager;
}

