/**
 * Dropdown Manager Module
 * =======================
 * Advanced dropdown management with positioning and keyboard navigation
 */

const DropdownManager = {
    /**
     * Active dropdowns
     */
    dropdowns: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        position: 'bottom-left',
        trigger: 'click',
        closeOnSelect: true,
        closeOnOutsideClick: true,
        keyboardNavigation: true,
        animation: true
    },
    
    /**
     * Initialize dropdown manager
     */
    init() {
        // Setup outside click handler
        document.addEventListener('click', (e) => {
            this.dropdowns.forEach((dropdownInfo) => {
                if (dropdownInfo.options.closeOnOutsideClick) {
                    const isClickInside = dropdownInfo.element.contains(e.target) || 
                                        dropdownInfo.trigger.contains(e.target);
                    if (!isClickInside) {
                        this.hide(dropdownInfo.id);
                    }
                }
            });
        });
        
        // Setup escape key handler
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.dropdowns.forEach((dropdownInfo) => {
                    this.hide(dropdownInfo.id);
                });
            }
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Dropdown manager initialized');
        }
    },
    
    /**
     * Show dropdown
     */
    show(trigger, menu, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const dropdownId = `dropdown_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create dropdown element
        const dropdown = this.createDropdownElement(dropdownId, menu, config);
        document.body.appendChild(dropdown);
        
        // Position dropdown
        this.positionDropdown(trigger, dropdown, config);
        
        // Store dropdown info
        const dropdownInfo = {
            id: dropdownId,
            element: dropdown,
            trigger: trigger,
            options: config,
            menu: menu
        };
        
        this.dropdowns.set(dropdownId, dropdownInfo);
        
        // Setup keyboard navigation
        if (config.keyboardNavigation) {
            this.setupKeyboardNavigation(dropdownInfo);
        }
        
        // Animate in
        if (config.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeIn(dropdown, 200);
        } else {
            dropdown.style.display = 'block';
        }
        
        // Emit dropdown show event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('dropdown:show', dropdownInfo);
        }
        
        return dropdownId;
    },
    
    /**
     * Create dropdown element
     */
    createDropdownElement(id, menu, options) {
        const dropdown = document.createElement('div');
        dropdown.id = id;
        dropdown.className = 'dropdown-menu';
        dropdown.setAttribute('role', 'menu');
        
        if (typeof menu === 'string') {
            dropdown.innerHTML = menu;
        } else if (menu instanceof HTMLElement) {
            dropdown.appendChild(menu);
        } else if (Array.isArray(menu)) {
            menu.forEach(item => {
                const menuItem = this.createMenuItem(item);
                dropdown.appendChild(menuItem);
            });
        }
        
        // Setup menu item clicks
        dropdown.querySelectorAll('[role="menuitem"]').forEach(item => {
            item.addEventListener('click', (e) => {
                if (options.closeOnSelect) {
                    this.hide(id);
                }
                
                // Emit menu item click event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('dropdown:item:click', { id, item, event: e });
                }
            });
        });
        
        return dropdown;
    },
    
    /**
     * Create menu item
     */
    createMenuItem(item) {
        const menuItem = document.createElement('div');
        menuItem.className = 'dropdown-item';
        menuItem.setAttribute('role', 'menuitem');
        menuItem.tabIndex = 0;
        
        if (typeof item === 'string') {
            menuItem.textContent = item;
        } else if (item.html) {
            menuItem.innerHTML = item.html;
        } else if (item.text) {
            menuItem.textContent = item.text;
        }
        
        if (item.onClick) {
            menuItem.addEventListener('click', item.onClick);
        }
        
        if (item.disabled) {
            menuItem.classList.add('disabled');
            menuItem.setAttribute('aria-disabled', 'true');
        }
        
        return menuItem;
    },
    
    /**
     * Position dropdown
     */
    positionDropdown(trigger, dropdown, options) {
        const triggerRect = trigger.getBoundingClientRect();
        const dropdownRect = dropdown.getBoundingClientRect();
        const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
        const scrollY = window.pageYOffset || document.documentElement.scrollTop;
        
        let top, left;
        
        const [vertical, horizontal] = options.position.split('-');
        
        // Vertical positioning
        if (vertical === 'top') {
            top = triggerRect.top + scrollY - dropdownRect.height;
        } else {
            top = triggerRect.bottom + scrollY;
        }
        
        // Horizontal positioning
        if (horizontal === 'left') {
            left = triggerRect.left + scrollX;
        } else if (horizontal === 'right') {
            left = triggerRect.right + scrollX - dropdownRect.width;
        } else {
            left = triggerRect.left + scrollX + (triggerRect.width / 2) - (dropdownRect.width / 2);
        }
        
        // Keep dropdown within viewport
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        if (left < 0) {
            left = 10;
        } else if (left + dropdownRect.width > viewportWidth) {
            left = viewportWidth - dropdownRect.width - 10;
        }
        
        if (top < 0) {
            top = 10;
        } else if (top + dropdownRect.height > viewportHeight) {
            top = viewportHeight - dropdownRect.height - 10;
        }
        
        dropdown.style.position = 'absolute';
        dropdown.style.top = `${top}px`;
        dropdown.style.left = `${left}px`;
    },
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation(dropdownInfo) {
        const items = dropdownInfo.element.querySelectorAll('[role="menuitem"]:not(.disabled)');
        
        if (items.length === 0) {
            return;
        }
        
        let currentIndex = 0;
        
        dropdownInfo.element.addEventListener('keydown', (e) => {
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    currentIndex = (currentIndex + 1) % items.length;
                    items[currentIndex].focus();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    currentIndex = (currentIndex - 1 + items.length) % items.length;
                    items[currentIndex].focus();
                    break;
                case 'Home':
                    e.preventDefault();
                    currentIndex = 0;
                    items[currentIndex].focus();
                    break;
                case 'End':
                    e.preventDefault();
                    currentIndex = items.length - 1;
                    items[currentIndex].focus();
                    break;
                case 'Enter':
                case ' ':
                    e.preventDefault();
                    items[currentIndex].click();
                    break;
            }
        });
        
        // Focus first item
        items[0].focus();
    },
    
    /**
     * Hide dropdown
     */
    hide(dropdownId) {
        const dropdownInfo = this.dropdowns.get(dropdownId);
        if (!dropdownInfo) {
            return false;
        }
        
        // Animate out
        if (dropdownInfo.options.animation && typeof AnimationManager !== 'undefined') {
            AnimationManager.fadeOut(dropdownInfo.element, 200).then(() => {
                this.removeDropdown(dropdownInfo);
            });
        } else {
            this.removeDropdown(dropdownInfo);
        }
        
        // Emit dropdown hide event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('dropdown:hide', dropdownInfo);
        }
        
        return true;
    },
    
    /**
     * Remove dropdown
     */
    removeDropdown(dropdownInfo) {
        if (dropdownInfo.element && dropdownInfo.element.parentNode) {
            dropdownInfo.element.parentNode.removeChild(dropdownInfo.element);
        }
        this.dropdowns.delete(dropdownInfo.id);
    },
    
    /**
     * Attach dropdown to element
     */
    attach(trigger, menu, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        let dropdownId = null;
        
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            if (dropdownId) {
                this.hide(dropdownId);
                dropdownId = null;
            } else {
                dropdownId = this.show(trigger, menu, config);
            }
        });
    },
    
    /**
     * Hide all dropdowns
     */
    hideAll() {
        const dropdownIds = Array.from(this.dropdowns.keys());
        dropdownIds.forEach(id => this.hide(id));
    },
    
    /**
     * Get dropdown
     */
    get(dropdownId) {
        return this.dropdowns.get(dropdownId);
    },
    
    /**
     * Get all dropdowns
     */
    getAll() {
        return Array.from(this.dropdowns.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    DropdownManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DropdownManager;
}

