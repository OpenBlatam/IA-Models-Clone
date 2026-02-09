/**
 * Accessibility Manager Module
 * ============================
 * Accessibility features and ARIA support
 */

const AccessibilityManager = {
    /**
     * Initialize accessibility features
     */
    init() {
        this.setupKeyboardNavigation();
        this.setupARIA();
        this.setupFocusManagement();
        this.detectScreenReader();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Accessibility manager initialized');
        }
    },
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation() {
        // Ensure all interactive elements are keyboard accessible
        document.addEventListener('keydown', (e) => {
            // Tab navigation enhancement
            if (e.key === 'Tab') {
                this.enhanceTabNavigation(e);
            }
            
            // Enter/Space for buttons
            if ((e.key === 'Enter' || e.key === ' ') && e.target.tagName === 'BUTTON') {
                e.preventDefault();
                e.target.click();
            }
        });
    },
    
    /**
     * Enhance tab navigation
     */
    enhanceTabNavigation(e) {
        const focusableElements = this.getFocusableElements();
        const currentIndex = focusableElements.indexOf(document.activeElement);
        
        // Skip hidden elements
        if (e.shiftKey && currentIndex > 0) {
            // Shift+Tab: go to previous
            for (let i = currentIndex - 1; i >= 0; i--) {
                if (this.isElementVisible(focusableElements[i])) {
                    focusableElements[i].focus();
                    break;
                }
            }
        } else if (!e.shiftKey && currentIndex < focusableElements.length - 1) {
            // Tab: go to next
            for (let i = currentIndex + 1; i < focusableElements.length; i++) {
                if (this.isElementVisible(focusableElements[i])) {
                    focusableElements[i].focus();
                    break;
                }
            }
        }
    },
    
    /**
     * Get focusable elements
     */
    getFocusableElements() {
        const selector = 'a[href], button, input, textarea, select, [tabindex]:not([tabindex="-1"])';
        return Array.from(document.querySelectorAll(selector));
    },
    
    /**
     * Check if element is visible
     */
    isElementVisible(element) {
        const style = window.getComputedStyle(element);
        return style.display !== 'none' && 
               style.visibility !== 'hidden' && 
               style.opacity !== '0';
    },
    
    /**
     * Setup ARIA attributes
     */
    setupARIA() {
        // Add ARIA labels to buttons without text
        document.querySelectorAll('button:not([aria-label]):empty').forEach(button => {
            const icon = button.textContent || button.innerHTML;
            if (icon && !button.getAttribute('aria-label')) {
                button.setAttribute('aria-label', `Button ${icon}`);
            }
        });
        
        // Add ARIA labels to form inputs
        document.querySelectorAll('input, textarea, select').forEach(input => {
            if (!input.getAttribute('aria-label') && !input.getAttribute('aria-labelledby')) {
                const label = document.querySelector(`label[for="${input.id}"]`);
                if (label) {
                    input.setAttribute('aria-labelledby', label.id || input.id + '-label');
                }
            }
        });
    },
    
    /**
     * Setup focus management
     */
    setupFocusManagement() {
        // Trap focus in modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.querySelector('.modal.show, [role="dialog"][aria-hidden="false"]');
                if (modal) {
                    this.closeModal(modal);
                }
            }
        });
    },
    
    /**
     * Close modal and restore focus
     */
    closeModal(modal) {
        const previousFocus = modal.getAttribute('data-previous-focus');
        modal.setAttribute('aria-hidden', 'true');
        modal.classList.remove('show');
        
        if (previousFocus) {
            const element = document.querySelector(`[data-focus-id="${previousFocus}"]`);
            if (element) {
                element.focus();
            }
        }
    },
    
    /**
     * Detect screen reader
     */
    detectScreenReader() {
        // Simple detection based on user agent
        const ua = navigator.userAgent.toLowerCase();
        const isScreenReader = ua.includes('nvda') || 
                              ua.includes('jaws') || 
                              ua.includes('voiceover');
        
        if (isScreenReader && typeof EventBus !== 'undefined') {
            EventBus.emit('accessibility:screen-reader:detected');
        }
        
        return isScreenReader;
    },
    
    /**
     * Announce to screen readers
     */
    announce(message, priority = 'polite') {
        const announcement = document.createElement('div');
        announcement.setAttribute('role', 'status');
        announcement.setAttribute('aria-live', priority);
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AccessibilityManager;
}

