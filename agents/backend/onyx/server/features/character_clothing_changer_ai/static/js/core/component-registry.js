/**
 * Component Registry Module
 * ==========================
 * Manages component registration and lifecycle
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
     * Initialize component registry
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Component registry initialized');
        }
        
        // Emit registry ready event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('component:registry_ready');
        }
    },
    
    /**
     * Register a component
     */
    register(name, component) {
        if (!component || typeof component !== 'object') {
            throw new Error(`Invalid component: ${name}`);
        }
        
        if (!component.name) {
            component.name = name;
        }
        
        // Validate component
        this.validateComponent(component);
        
        this.components.set(name, component);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Component registered: ${name}`);
        }
        
        // Emit component registered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('component:registered', { name, component });
        }
        
        return this;
    },
    
    /**
     * Unregister a component
     */
    unregister(name) {
        const component = this.components.get(name);
        if (!component) {
            return false;
        }
        
        // Destroy all instances
        const instances = Array.from(this.instances.entries())
            .filter(([_, instance]) => instance.component === name);
        
        instances.forEach(([instanceId]) => {
            this.destroy(instanceId);
        });
        
        this.components.delete(name);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Component unregistered: ${name}`);
        }
        
        // Emit component unregistered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('component:unregistered', { name });
        }
        
        return true;
    },
    
    /**
     * Create component instance
     */
    create(name, container, props = {}) {
        const component = this.components.get(name);
        if (!component) {
            throw new Error(`Component not found: ${name}`);
        }
        
        const instanceId = `${name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create instance
        const instance = {
            id: instanceId,
            component: name,
            container,
            props,
            element: null,
            state: {}
        };
        
        // Initialize component if init method exists
        if (typeof component.init === 'function') {
            try {
                instance.element = component.init(container, props, instance.state);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error initializing component ${name}:`, error);
                }
                throw error;
            }
        } else if (typeof component.render === 'function') {
            // Render component
            instance.element = component.render(container, props, instance.state);
        } else {
            throw new Error(`Component ${name} has no init or render method`);
        }
        
        // Mount element if container provided
        if (container && instance.element) {
            if (typeof container.appendChild === 'function') {
                container.appendChild(instance.element);
            } else if (typeof container === 'string') {
                const containerEl = document.querySelector(container);
                if (containerEl) {
                    containerEl.appendChild(instance.element);
                }
            }
        }
        
        this.instances.set(instanceId, instance);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Component instance created: ${name}`, { id: instanceId });
        }
        
        // Emit instance created event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('component:instance_created', { name, instanceId });
        }
        
        return instanceId;
    },
    
    /**
     * Destroy component instance
     */
    destroy(instanceId) {
        const instance = this.instances.get(instanceId);
        if (!instance) {
            return false;
        }
        
        const component = this.components.get(instance.component);
        
        // Cleanup component if cleanup method exists
        if (component && typeof component.cleanup === 'function') {
            try {
                component.cleanup(instance.element, instance.props, instance.state);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error cleaning up component ${instance.component}:`, error);
                }
            }
        }
        
        // Remove element from DOM
        if (instance.element && instance.element.parentNode) {
            instance.element.parentNode.removeChild(instance.element);
        }
        
        this.instances.delete(instanceId);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Component instance destroyed: ${instance.component}`, { id: instanceId });
        }
        
        // Emit instance destroyed event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('component:instance_destroyed', { instanceId });
        }
        
        return true;
    },
    
    /**
     * Get component
     */
    get(name) {
        return this.components.get(name);
    },
    
    /**
     * Get component instance
     */
    getInstance(instanceId) {
        return this.instances.get(instanceId);
    },
    
    /**
     * Get all components
     */
    getAll() {
        return Array.from(this.components.values());
    },
    
    /**
     * Get all instances
     */
    getAllInstances() {
        return Array.from(this.instances.values());
    },
    
    /**
     * Get instances by component name
     */
    getInstancesByComponent(name) {
        return Array.from(this.instances.values())
            .filter(instance => instance.component === name);
    },
    
    /**
     * Validate component
     */
    validateComponent(component) {
        if (!component.name) {
            throw new Error('Component must have a name');
        }
        
        if (typeof component.init !== 'function' && typeof component.render !== 'function') {
            throw new Error('Component must have init or render method');
        }
        
        return true;
    },
    
    /**
     * Update component instance
     */
    update(instanceId, newProps) {
        const instance = this.instances.get(instanceId);
        if (!instance) {
            return false;
        }
        
        const component = this.components.get(instance.component);
        if (!component) {
            return false;
        }
        
        // Update props
        instance.props = { ...instance.props, ...newProps };
        
        // Re-render if update method exists
        if (typeof component.update === 'function') {
            try {
                component.update(instance.element, instance.props, instance.state);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error updating component ${instance.component}:`, error);
                }
            }
        } else if (typeof component.render === 'function') {
            // Re-render
            const newElement = component.render(instance.container, instance.props, instance.state);
            if (instance.element && instance.element.parentNode) {
                instance.element.parentNode.replaceChild(newElement, instance.element);
            }
            instance.element = newElement;
        }
        
        return true;
    },
    
    /**
     * Clear all components
     */
    clear() {
        // Destroy all instances
        const instanceIds = Array.from(this.instances.keys());
        instanceIds.forEach(id => this.destroy(id));
        
        // Clear components
        this.components.clear();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Component registry cleared');
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ComponentRegistry;
}

