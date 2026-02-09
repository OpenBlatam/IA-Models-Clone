/**
 * Resource Manager Module
 * ======================
 * Resource loading and management
 */

const ResourceManager = {
    /**
     * Loaded resources
     */
    resources: new Map(),
    
    /**
     * Loading queue
     */
    queue: [],
    
    /**
     * Load a resource
     */
    async load(type, url, options = {}) {
        const cacheKey = `${type}:${url}`;
        
        // Check cache
        if (this.resources.has(cacheKey)) {
            return this.resources.get(cacheKey);
        }
        
        // Check Cache module
        if (typeof Cache !== 'undefined') {
            const cached = Cache.get(cacheKey);
            if (cached) {
                this.resources.set(cacheKey, cached);
                return cached;
            }
        }
        
        try {
            let resource;
            
            switch (type) {
                case 'script':
                    resource = await this.loadScript(url, options);
                    break;
                case 'style':
                    resource = await this.loadStyle(url, options);
                    break;
                case 'image':
                    resource = await this.loadImage(url, options);
                    break;
                case 'json':
                    resource = await this.loadJSON(url, options);
                    break;
                default:
                    resource = await fetch(url).then(r => r.text());
            }
            
            // Cache resource
            this.resources.set(cacheKey, resource);
            
            if (typeof Cache !== 'undefined') {
                Cache.set(cacheKey, resource, options.ttl);
            }
            
            // Emit resource loaded event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('resource:loaded', { type, url, resource });
            }
            
            return resource;
        } catch (error) {
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handle(error, { context: `resource loading: ${type}`, url });
            }
            throw error;
        }
    },
    
    /**
     * Load script
     */
    async loadScript(url, options = {}) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.async = options.async !== false;
            script.defer = options.defer || false;
            
            script.onload = () => resolve(script);
            script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
            
            document.head.appendChild(script);
        });
    },
    
    /**
     * Load style
     */
    async loadStyle(url, options = {}) {
        return new Promise((resolve, reject) => {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = url;
            
            link.onload = () => resolve(link);
            link.onerror = () => reject(new Error(`Failed to load style: ${url}`));
            
            document.head.appendChild(link);
        });
    },
    
    /**
     * Load image
     */
    async loadImage(url, options = {}) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = url;
            
            if (options.crossOrigin) {
                img.crossOrigin = options.crossOrigin;
            }
            
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
        });
    },
    
    /**
     * Load JSON
     */
    async loadJSON(url, options = {}) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load JSON: ${url}`);
        }
        return response.json();
    },
    
    /**
     * Preload resources
     */
    async preload(resources) {
        const promises = resources.map(resource => 
            this.load(resource.type, resource.url, resource.options)
        );
        
        return Promise.all(promises);
    },
    
    /**
     * Clear resources
     */
    clear() {
        this.resources.clear();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResourceManager;
}

