/**
 * Accessibility Manager Module
 * ============================
 * Manages accessibility features and ARIA attributes
 */

const AccessibilityManager = {
    /**
     * Accessibility settings
     */
    settings: {
        highContrast: false,
        reducedMotion: false,
        fontSize: 'medium',
        screenReader: false
    },
    
    /**
     * Initialize accessibility manager
     */
    init() {
        // Load settings from storage
        this.loadSettings();
        
        // Detect preferences
        this.detectPreferences();
        
        // Apply settings
        this.applySettings();
        
        // Setup keyboard navigation
        this.setupKeyboardNavigation();
        
        // Setup focus management
        this.setupFocusManagement();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Accessibility manager initialized');
        }
    },
    
    /**
     * Load settings from storage
     */
    loadSettings() {
        try {
            const stored = localStorage.getItem('accessibility_settings');
            if (stored) {
                this.settings = { ...this.settings, ...JSON.parse(stored) };
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to load accessibility settings', error);
            }
        }
    },
    
    /**
     * Save settings to storage
     */
    saveSettings() {
        try {
            localStorage.setItem('accessibility_settings', JSON.stringify(this.settings));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save accessibility settings', error);
            }
        }
    },
    
    /**
     * Detect user preferences
     */
    detectPreferences() {
        // Detect reduced motion preference
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            this.settings.reducedMotion = true;
        }
        
        // Detect high contrast preference
        if (window.matchMedia('(prefers-contrast: high)').matches) {
            this.settings.highContrast = true;
        }
        
        // Detect screen reader (basic detection)
        if (navigator.userAgent.includes('NVDA') || 
            navigator.userAgent.includes('JAWS') ||
            navigator.userAgent.includes('VoiceOver')) {
            this.settings.screenReader = true;
        }
    },
    
    /**
     * Apply accessibility settings
     */
    applySettings() {
        const root = document.documentElement;
        
        // High contrast
        if (this.settings.highContrast) {
            root.classList.add('high-contrast');
        } else {
            root.classList.remove('high-contrast');
        }
        
        // Reduced motion
        if (this.settings.reducedMotion) {
            root.classList.add('reduced-motion');
        } else {
            root.classList.remove('reduced-motion');
        }
        
        // Font size
        root.setAttribute('data-font-size', this.settings.fontSize);
        
        // Screen reader
        if (this.settings.screenReader) {
            root.classList.add('screen-reader-mode');
        } else {
            root.classList.remove('screen-reader-mode');
        }
    },
    
    /**
     * Set high contrast
     */
    setHighContrast(enabled) {
        this.settings.highContrast = enabled;
        this.applySettings();
        this.saveSettings();
    },
    
    /**
     * Set reduced motion
     */
    setReducedMotion(enabled) {
        this.settings.reducedMotion = enabled;
        this.applySettings();
        this.saveSettings();
    },
    
    /**
     * Set font size
     */
    setFontSize(size) {
        const validSizes = ['small', 'medium', 'large', 'xlarge'];
        if (validSizes.includes(size)) {
            this.settings.fontSize = size;
            this.applySettings();
            this.saveSettings();
        }
    },
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation() {
        // Skip links
        this.createSkipLinks();
        
        // Focus trap for modals
        this.setupFocusTrap();
        
        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
    },
    
    /**
     * Create skip links
     */
    createSkipLinks() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link';
        skipLink.textContent = 'Saltar al contenido principal';
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 0;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
            z-index: 100;
        `;
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '0';
        });
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });
        document.body.insertBefore(skipLink, document.body.firstChild);
    },
    
    /**
     * Setup focus trap
     */
    setupFocusTrap() {
        // This would be implemented when modals are opened
        // Focus trap logic goes here
    },
    
    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Alt + S: Skip to main content
            if (e.altKey && e.key === 's') {
                e.preventDefault();
                const main = document.querySelector('main') || document.querySelector('#main-content');
                if (main) {
                    main.focus();
                    main.scrollIntoView();
                }
            }
        });
    },
    
    /**
     * Setup focus management
     */
    setupFocusManagement() {
        // Track focus changes
        document.addEventListener('focusin', (e) => {
            // Add focus indicator
            e.target.classList.add('keyboard-focus');
        });
        
        document.addEventListener('focusout', (e) => {
            // Remove focus indicator
            e.target.classList.remove('keyboard-focus');
        });
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
    },
    
    /**
     * Get settings
     */
    getSettings() {
        return { ...this.settings };
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    AccessibilityManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AccessibilityManager;
}

