/**
 * Component Registry Module
 * =========================
 * Component registration and management system
 */

const ComponentRegistry = {
    /**
     * Registered components
     */
    components: new Map(),
    
    /**
     * Component instances
     */
    instances: new Map(),
    
    /**
     * Register a component
     */
    register(name, component) {
        if (!component.render || typeof component.render !== 'function') {
            throw new Error('Component must have a render function');
        }
        
        this.components.set(name, {
            ...component,
            registeredAt: Date.now()
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Component registered: ${name}`);
        }
        
        // Emit component registered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('component:registered', { name, component });
        }
        
        return true;
    },
    
    /**
     * Create component instance
     */
    create(name, props = {}, container = null) {
        const component = this.components.get(name);
        if (!component) {
            throw new Error(`Component not found: ${name}`);
        }
        
        const instance = {
            id: `${name}_${Date.now()}`,
            name,
            props,
            container,
            element: null,
            createdAt: Date.now()
        };
        
        // Render component
        instance.element = component.render(props, instance);
        
        // Mount to container if provided
        if (container) {
            if (typeof container === 'string') {
                container = document.querySelector(container);
            }
            if (container) {
                container.appendChild(instance.element);
            }
        }
        
        // Store instance
        this.instances.set(instance.id, instance);
        
        // Call mounted hook if available
        if (component.mounted) {
            component.mounted(instance);
        }
        
        return instance;
    },
    
    /**
     * Destroy component instance
     */
    destroy(instanceId) {
        const instance = this.instances.get(instanceId);
        if (!instance) {
            return false;
        }
        
        // Get component
        const component = this.components.get(instance.name);
        
        // Call unmount hook if available
        if (component && component.unmounted) {
            component.unmounted(instance);
        }
        
        // Remove from DOM
        if (instance.element && instance.element.parentNode) {
            instance.element.parentNode.removeChild(instance.element);
        }
        
        // Remove instance
        this.instances.delete(instanceId);
        
        return true;
    },
    
    /**
     * Get component
     */
    getComponent(name) {
        return this.components.get(name);
    },
    
    /**
     * Get instance
     */
    getInstance(instanceId) {
        return this.instances.get(instanceId);
    },
    
    /**
     * Get all components
     */
    getAllComponents() {
        return Array.from(this.components.keys());
    },
    
    /**
     * Get all instances
     */
    getAllInstances() {
        return Array.from(this.instances.values());
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ComponentRegistry;
}

