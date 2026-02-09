/**
 * Example Plugin
 * =============
 * Example plugin demonstrating the plugin system
 */

const ExamplePlugin = {
    name: 'example-plugin',
    version: '1.0.0',
    description: 'Example plugin for demonstration',
    author: 'Your Name',
    enabled: true,
    
    /**
     * Initialize plugin
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Example plugin initialized');
        }
        
        // Register hooks
        this.hooks = {
            'form:before_submit': this.beforeFormSubmit.bind(this),
            'form:after_submit': this.afterFormSubmit.bind(this),
            'result:display': this.onResultDisplay.bind(this)
        };
    },
    
    /**
     * Cleanup plugin
     */
    cleanup() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Example plugin cleaned up');
        }
    },
    
    /**
     * Hook: Before form submit
     */
    beforeFormSubmit(formData) {
        // Modify form data before submission
        if (typeof Logger !== 'undefined') {
            Logger.debug('Example plugin: before form submit', formData);
        }
        
        // Example: Add custom field
        // formData.append('custom_field', 'custom_value');
        
        return formData;
    },
    
    /**
     * Hook: After form submit
     */
    afterFormSubmit(result) {
        // Process result after submission
        if (typeof Logger !== 'undefined') {
            Logger.debug('Example plugin: after form submit', result);
        }
        
        // Example: Add custom processing
        // result.customProperty = 'customValue';
        
        return result;
    },
    
    /**
     * Hook: On result display
     */
    onResultDisplay(resultHTML) {
        // Modify result HTML before display
        if (typeof Logger !== 'undefined') {
            Logger.debug('Example plugin: on result display');
        }
        
        // Example: Add custom element
        // resultHTML += '<div class="custom-element">Custom content</div>';
        
        return resultHTML;
    }
};

// Auto-register if PluginManager is available
if (typeof PluginManager !== 'undefined') {
    PluginManager.register('example-plugin', ExamplePlugin);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExamplePlugin;
}

