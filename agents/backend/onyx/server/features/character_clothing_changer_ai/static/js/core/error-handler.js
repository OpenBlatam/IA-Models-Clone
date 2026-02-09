/**
 * Error Handler Module
 * ====================
 * Centralized error handling and reporting
 */

const ErrorHandler = {
    /**
     * Handle error
     */
    handle(error, context = {}) {
        const errorInfo = {
            message: error.message || error.toString(),
            stack: error.stack,
            context,
            timestamp: new Date().toISOString()
        };
        
        // Log error
        this.log(errorInfo);
        
        // Show user-friendly notification
        this.notifyUser(error, context);
        
        // Report to server if needed
        if (this.shouldReport(error)) {
            this.report(errorInfo);
        }
    },

    /**
     * Log error
     */
    log(errorInfo) {
        console.error('Error:', errorInfo);
        
        // Store in localStorage for debugging
        try {
            const errors = JSON.parse(localStorage.getItem('app_errors') || '[]');
            errors.push(errorInfo);
            
            // Keep only last 10 errors
            if (errors.length > 10) {
                errors.shift();
            }
            
            localStorage.setItem('app_errors', JSON.stringify(errors));
        } catch (e) {
            console.error('Failed to store error:', e);
        }
    },

    /**
     * Notify user about error
     */
    notifyUser(error, context) {
        if (typeof Notifications === 'undefined') return;
        
        let message = 'Ha ocurrido un error';
        
        // Customize message based on error type
        if (error.message) {
            if (error.message.includes('network') || error.message.includes('fetch')) {
                message = 'Error de conexión. Verifica tu internet.';
            } else if (error.message.includes('validation')) {
                message = 'Error de validación. Revisa los datos ingresados.';
            } else if (error.message.includes('timeout')) {
                message = 'Tiempo de espera agotado. Intenta nuevamente.';
            } else {
                message = error.message;
            }
        }
        
        Notifications.error(message, 5000);
    },

    /**
     * Determine if error should be reported to server
     */
    shouldReport(error) {
        // Report critical errors
        const criticalPatterns = [
            'Failed to build',
            'Cannot load model',
            'Out of memory',
            'Internal server error'
        ];
        
        return criticalPatterns.some(pattern => 
            error.message && error.message.includes(pattern)
        );
    },

    /**
     * Report error to server
     */
    async report(errorInfo) {
        try {
            if (typeof API === 'undefined') return;
            
            // Only report in production or if explicitly enabled
            if (CONFIG && CONFIG.ERROR_REPORTING) {
                await API.reportError(errorInfo);
            }
        } catch (e) {
            console.error('Failed to report error:', e);
        }
    },

    /**
     * Get error history
     */
    getErrorHistory() {
        try {
            return JSON.parse(localStorage.getItem('app_errors') || '[]');
        } catch (e) {
            return [];
        }
    },

    /**
     * Clear error history
     */
    clearErrorHistory() {
        localStorage.removeItem('app_errors');
    },

    /**
     * Wrap async function with error handling
     */
    wrapAsync(fn) {
        return async (...args) => {
            try {
                return await fn(...args);
            } catch (error) {
                this.handle(error, { function: fn.name, args });
                throw error;
            }
        };
    },

    /**
     * Format error message for display
     */
    formatError(error) {
        if (error.message) {
            return error.message;
        }
        return error.toString();
    },

    /**
     * Handle API error specifically
     */
    handleApiError(error, context) {
        this.handle(error, { context, type: 'api' });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ErrorHandler;
}

