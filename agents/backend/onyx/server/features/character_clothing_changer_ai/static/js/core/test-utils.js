/**
 * Test Utils Module
 * =================
 * Utilities for testing and development
 */

const TestUtils = {
    /**
     * Mock API responses
     */
    mockAPI: {
        responses: new Map(),
        
        /**
         * Set mock response
         */
        set(endpoint, response) {
            this.responses.set(endpoint, response);
        },
        
        /**
         * Get mock response
         */
        get(endpoint) {
            return this.responses.get(endpoint);
        },
        
        /**
         * Clear all mocks
         */
        clear() {
            this.responses.clear();
        }
    },
    
    /**
     * Create test data
     */
    createTestData(type = 'result') {
        const baseData = {
            id: `test_${Date.now()}`,
            timestamp: new Date().toISOString(),
            character_name: 'Test Character',
            clothing_description: 'test clothing',
            image_base64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
        };
        
        switch (type) {
            case 'result':
                return {
                    ...baseData,
                    changed: true,
                    quality_metrics: {
                        similarity: 0.95,
                        clarity: 0.90
                    }
                };
            case 'history':
                return baseData;
            case 'gallery':
                return {
                    ...baseData,
                    favorite: false
                };
            default:
                return baseData;
        }
    },
    
    /**
     * Mock localStorage
     */
    mockLocalStorage() {
        const store = {};
        return {
            getItem: (key) => store[key] || null,
            setItem: (key, value) => { store[key] = value; },
            removeItem: (key) => { delete store[key]; },
            clear: () => { Object.keys(store).forEach(key => delete store[key]); }
        };
    },
    
    /**
     * Wait for condition
     */
    async waitFor(condition, timeout = 5000, interval = 100) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            if (await condition()) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, interval));
        }
        
        throw new Error('Condition not met within timeout');
    },
    
    /**
     * Create mock event
     */
    createMockEvent(type, data = {}) {
        return {
            type,
            ...data,
            preventDefault: () => {},
            stopPropagation: () => {}
        };
    },
    
    /**
     * Test module initialization
     */
    async testModule(moduleName, initFn) {
        try {
            await initFn();
            if (typeof Logger !== 'undefined') {
                Logger.info(`Module test passed: ${moduleName}`);
            }
            return { success: true, module: moduleName };
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Module test failed: ${moduleName}`, error);
            }
            return { success: false, module: moduleName, error };
        }
    },
    
    /**
     * Run all tests
     */
    async runTests(tests) {
        const results = [];
        for (const test of tests) {
            const result = await this.testModule(test.name, test.fn);
            results.push(result);
        }
        return results;
    },
    
    /**
     * Performance test
     */
    async performanceTest(name, fn, iterations = 100) {
        const times = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await fn();
            const end = performance.now();
            times.push(end - start);
        }
        
        const avg = times.reduce((a, b) => a + b, 0) / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        
        return {
            name,
            iterations,
            average: avg,
            min,
            max,
            total: times.reduce((a, b) => a + b, 0)
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TestUtils;
}

