/**
 * Documentation Generator Module
 * ==============================
 * Auto-generate documentation from code
 */

const DocumentationGenerator = {
    /**
     * Generate module documentation
     */
    generateModuleDocs(moduleName, module) {
        const docs = {
            name: moduleName,
            description: module.description || 'No description available',
            methods: [],
            properties: [],
            events: [],
            examples: []
        };
        
        // Extract methods
        if (typeof module === 'object') {
            Object.keys(module).forEach(key => {
                if (typeof module[key] === 'function') {
                    docs.methods.push({
                        name: key,
                        signature: this.extractFunctionSignature(module[key])
                    });
                } else if (key !== 'description') {
                    docs.properties.push({
                        name: key,
                        type: typeof module[key]
                    });
                }
            });
        }
        
        return docs;
    },
    
    /**
     * Extract function signature
     */
    extractFunctionSignature(fn) {
        const str = fn.toString();
        const match = str.match(/\([^)]*\)/);
        return match ? match[0] : '()';
    },
    
    /**
     * Generate API documentation
     */
    generateAPIDocs() {
        const modules = {};
        
        // Collect all modules
        if (typeof window !== 'undefined') {
            const moduleNames = [
                'EventBus', 'StateManager', 'API', 'Form', 'GalleryManager',
                'HistoryManager', 'Notifications', 'Favorites', 'Filters',
                'PerformanceMonitor', 'Analytics', 'HealthMonitor'
            ];
            
            moduleNames.forEach(name => {
                if (typeof window[name] !== 'undefined') {
                    modules[name] = this.generateModuleDocs(name, window[name]);
                }
            });
        }
        
        return {
            version: '1.0.0',
            generatedAt: new Date().toISOString(),
            modules
        };
    },
    
    /**
     * Export documentation
     */
    exportDocs(format = 'json') {
        const docs = this.generateAPIDocs();
        
        if (format === 'json') {
            return JSON.stringify(docs, null, 2);
        } else if (format === 'markdown') {
            return this.generateMarkdown(docs);
        }
        
        return docs;
    },
    
    /**
     * Generate markdown documentation
     */
    generateMarkdown(docs) {
        let md = `# API Documentation\n\n`;
        md += `Generated at: ${docs.generatedAt}\n\n`;
        
        Object.values(docs.modules).forEach(module => {
            md += `## ${module.name}\n\n`;
            md += `${module.description}\n\n`;
            
            if (module.methods.length > 0) {
                md += `### Methods\n\n`;
                module.methods.forEach(method => {
                    md += `- **${method.name}**${method.signature}\n`;
                });
                md += `\n`;
            }
            
            if (module.properties.length > 0) {
                md += `### Properties\n\n`;
                module.properties.forEach(prop => {
                    md += `- **${prop.name}**: ${prop.type}\n`;
                });
                md += `\n`;
            }
        });
        
        return md;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DocumentationGenerator;
}

