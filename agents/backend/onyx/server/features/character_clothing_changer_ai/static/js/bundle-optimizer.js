/**
 * Bundle Optimizer Module
 * ======================
 * Optimize and manage JavaScript bundles
 */

const BundleOptimizer = {
    /**
     * Module sizes
     */
    moduleSizes: new Map(),
    
    /**
     * Loaded modules
     */
    loadedModules: new Set(),
    
    /**
     * Track module load
     */
    trackModuleLoad(moduleName, size) {
        this.moduleSizes.set(moduleName, size);
        this.loadedModules.add(moduleName);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Module loaded: ${moduleName} (${size} bytes)`);
        }
    },
    
    /**
     * Get bundle statistics
     */
    getStats() {
        const totalSize = Array.from(this.moduleSizes.values())
            .reduce((sum, size) => sum + size, 0);
        
        const modules = Array.from(this.moduleSizes.entries())
            .map(([name, size]) => ({
                name,
                size,
                percentage: (size / totalSize) * 100
            }))
            .sort((a, b) => b.size - a.size);
        
        return {
            totalModules: this.loadedModules.size,
            totalSize,
            totalSizeKB: (totalSize / 1024).toFixed(2),
            totalSizeMB: (totalSize / 1024 / 1024).toFixed(2),
            modules
        };
    },
    
    /**
     * Get largest modules
     */
    getLargestModules(limit = 10) {
        const stats = this.getStats();
        return stats.modules.slice(0, limit);
    },
    
    /**
     * Suggest optimizations
     */
    getOptimizationSuggestions() {
        const suggestions = [];
        const stats = this.getStats();
        
        // Check total size
        if (stats.totalSize > 1024 * 1024) { // > 1MB
            suggestions.push({
                type: 'warning',
                message: `Bundle size is ${stats.totalSizeMB}MB. Consider code splitting.`
            });
        }
        
        // Check for large modules
        const largeModules = stats.modules.filter(m => m.size > 50 * 1024); // > 50KB
        if (largeModules.length > 0) {
            suggestions.push({
                type: 'info',
                message: `${largeModules.length} modules are larger than 50KB. Consider lazy loading.`,
                modules: largeModules.map(m => m.name)
            });
        }
        
        return suggestions;
    },
    
    /**
     * Export bundle report
     */
    exportReport() {
        return JSON.stringify({
            stats: this.getStats(),
            suggestions: this.getOptimizationSuggestions(),
            timestamp: Date.now()
        }, null, 2);
    }
};

// Auto-track script loads
if (typeof window !== 'undefined' && typeof performance !== 'undefined') {
    const originalCreateElement = document.createElement;
    document.createElement = function(tagName) {
        const element = originalCreateElement.call(document, tagName);
        
        if (tagName === 'script' && element.src) {
            element.addEventListener('load', () => {
                // Try to get resource size from performance API
                const entries = performance.getEntriesByName(element.src, 'resource');
                if (entries.length > 0) {
                    const size = entries[0].transferSize || entries[0].decodedBodySize || 0;
                    const moduleName = element.src.split('/').pop().replace('.js', '');
                    BundleOptimizer.trackModuleLoad(moduleName, size);
                }
            });
        }
        
        return element;
    };
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BundleOptimizer;
}

