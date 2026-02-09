/**
 * Test Runner Module
 * ==================
 * Simple test framework for development
 */

const TestRunner = {
    /**
     * Test suite
     */
    tests: [],
    
    /**
     * Test results
     */
    results: [],
    
    /**
     * Register a test
     */
    test(name, fn) {
        this.tests.push({ name, fn });
    },
    
    /**
     * Assert helper
     */
    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    },
    
    /**
     * Run all tests
     */
    async run() {
        this.results = [];
        
        for (const test of this.tests) {
            try {
                await test.fn();
                this.results.push({
                    name: test.name,
                    status: 'passed',
                    error: null
                });
            } catch (error) {
                this.results.push({
                    name: test.name,
                    status: 'failed',
                    error: error.message
                });
            }
        }
        
        return this.getResults();
    },
    
    /**
     * Get test results
     */
    getResults() {
        const passed = this.results.filter(r => r.status === 'passed').length;
        const failed = this.results.filter(r => r.status === 'failed').length;
        
        return {
            total: this.results.length,
            passed,
            failed,
            results: this.results
        };
    },
    
    /**
     * Clear tests
     */
    clear() {
        this.tests = [];
        this.results = [];
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TestRunner;
}

