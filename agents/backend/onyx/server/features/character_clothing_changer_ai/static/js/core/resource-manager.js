/**
 * Resource Manager Module
 * =======================
 * Manages application resources (images, fonts, etc.)
 */

const ResourceManager = {
    /**
     * Loaded resources
     */
    resources: new Map(),
    
    /**
     * Loading promises
     */
    loadingPromises: new Map(),
    
    /**
     * Initialize resource manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Resource manager initialized');
        }
    },
    
    /**
     * Load image
     */
    async loadImage(src) {
        if (this.resources.has(src)) {
            return this.resources.get(src);
        }
        
        if (this.loadingPromises.has(src)) {
            return this.loadingPromises.get(src);
        }
        
        const promise = new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => {
                this.resources.set(src, img);
                this.loadingPromises.delete(src);
                resolve(img);
            };
            
            img.onerror = () => {
                this.loadingPromises.delete(src);
                reject(new Error(`Failed to load image: ${src}`));
            };
            
            img.src = src;
        });
        
        this.loadingPromises.set(src, promise);
        return promise;
    },
    
    /**
     * Preload images
     */
    async preloadImages(srcs) {
        const promises = srcs.map(src => 
            this.loadImage(src).catch(error => {
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Failed to preload image: ${src}`, error);
                }
                return null;
            })
        );
        
        return Promise.all(promises);
    },
    
    /**
     * Load font
     */
    async loadFont(family, src, options = {}) {
        const key = `${family}-${src}`;
        
        if (this.resources.has(key)) {
            return this.resources.get(key);
        }
        
        return new Promise((resolve, reject) => {
            const font = new FontFace(family, `url(${src})`, options);
            
            font.load().then(() => {
                document.fonts.add(font);
                this.resources.set(key, font);
                resolve(font);
            }).catch(reject);
        });
    },
    
    /**
     * Load script
     */
    async loadScript(src, options = {}) {
        if (this.resources.has(src)) {
            return this.resources.get(src);
        }
        
        if (this.loadingPromises.has(src)) {
            return this.loadingPromises.get(src);
        }
        
        const promise = new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.async = options.async !== false;
            script.defer = options.defer || false;
            
            script.onload = () => {
                this.resources.set(src, script);
                this.loadingPromises.delete(src);
                resolve(script);
            };
            
            script.onerror = () => {
                this.loadingPromises.delete(src);
                reject(new Error(`Failed to load script: ${src}`));
            };
            
            document.head.appendChild(script);
        });
        
        this.loadingPromises.set(src, promise);
        return promise;
    },
    
    /**
     * Load stylesheet
     */
    async loadStylesheet(href) {
        if (this.resources.has(href)) {
            return this.resources.get(href);
        }
        
        if (this.loadingPromises.has(href)) {
            return this.loadingPromises.get(href);
        }
        
        const promise = new Promise((resolve, reject) => {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = href;
            
            link.onload = () => {
                this.resources.set(href, link);
                this.loadingPromises.delete(href);
                resolve(link);
            };
            
            link.onerror = () => {
                this.loadingPromises.delete(href);
                reject(new Error(`Failed to load stylesheet: ${href}`));
            };
            
            document.head.appendChild(link);
        });
        
        this.loadingPromises.set(href, promise);
        return promise;
    },
    
    /**
     * Get resource
     */
    get(src) {
        return this.resources.get(src);
    },
    
    /**
     * Check if resource is loaded
     */
    isLoaded(src) {
        return this.resources.has(src);
    },
    
    /**
     * Clear resources
     */
    clear() {
        this.resources.clear();
        this.loadingPromises.clear();
    },
    
    /**
     * Get resource stats
     */
    getStats() {
        return {
            total: this.resources.size,
            loading: this.loadingPromises.size,
            types: this.getResourceTypes()
        };
    },
    
    /**
     * Get resource types
     */
    getResourceTypes() {
        const types = {};
        this.resources.forEach((resource, key) => {
            const type = resource.constructor.name;
            types[type] = (types[type] || 0) + 1;
        });
        return types;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResourceManager;
}

