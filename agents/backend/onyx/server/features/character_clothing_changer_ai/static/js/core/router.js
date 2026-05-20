/**
 * Router Module
 * =============
 * Client-side routing system
 */

const Router = {
    /**
     * Routes
     */
    routes: new Map(),
    
    /**
     * Current route
     */
    currentRoute: null,
    
    /**
     * Route history
     */
    history: [],
    
    /**
     * Initialize router
     */
    init() {
        // Listen to popstate (browser back/forward)
        window.addEventListener('popstate', (e) => {
            this.handleRoute(window.location.pathname);
        });
        
        // Handle initial route
        this.handleRoute(window.location.pathname);
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Router initialized');
        }
    },
    
    /**
     * Register route
     */
    route(path, handler, options = {}) {
        const route = {
            path,
            handler,
            name: options.name || path,
            middleware: options.middleware || [],
            meta: options.meta || {}
        };
        
        this.routes.set(path, route);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Route registered: ${path}`);
        }
        
        return this;
    },
    
    /**
     * Navigate to route
     */
    navigate(path, options = {}) {
        const { replace = false, state = {} } = options;
        
        if (replace) {
            window.history.replaceState(state, '', path);
        } else {
            window.history.pushState(state, '', path);
        }
        
        this.handleRoute(path);
    },
    
    /**
     * Handle route
     */
    async handleRoute(path) {
        const route = this.findRoute(path);
        
        if (!route) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Route not found: ${path}`);
            }
            return;
        }
        
        // Execute middleware
        for (const middleware of route.middleware) {
            const result = await middleware(path, route);
            if (result === false) {
                // Middleware blocked navigation
                return;
            }
        }
        
        // Execute route handler
        try {
            await route.handler(path, route);
            
            // Update current route
            this.currentRoute = route;
            
            // Add to history
            this.history.push({
                path,
                route: route.name,
                timestamp: Date.now()
            });
            
            // Keep only last 50 entries
            if (this.history.length > 50) {
                this.history.shift();
            }
            
            // Emit route changed event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('route:changed', { path, route });
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Route handler error for ${path}:`, error);
            }
            
            // Emit route error event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('route:error', { path, route, error });
            }
        }
    },
    
    /**
     * Find route for path
     */
    findRoute(path) {
        // Exact match
        if (this.routes.has(path)) {
            return this.routes.get(path);
        }
        
        // Pattern match
        for (const [routePath, route] of this.routes.entries()) {
            if (this.matchRoute(routePath, path)) {
                return route;
            }
        }
        
        return null;
    },
    
    /**
     * Match route pattern
     */
    matchRoute(pattern, path) {
        // Convert pattern to regex
        const regex = new RegExp('^' + pattern.replace(/:\w+/g, '([^/]+)') + '$');
        return regex.test(path);
    },
    
    /**
     * Get route params
     */
    getRouteParams(pattern, path) {
        const params = {};
        const patternParts = pattern.split('/');
        const pathParts = path.split('/');
        
        patternParts.forEach((part, index) => {
            if (part.startsWith(':')) {
                const paramName = part.slice(1);
                params[paramName] = pathParts[index];
            }
        });
        
        return params;
    },
    
    /**
     * Get current route
     */
    getCurrentRoute() {
        return this.currentRoute;
    },
    
    /**
     * Get route history
     */
    getHistory() {
        return [...this.history];
    },
    
    /**
     * Go back
     */
    back() {
        window.history.back();
    },
    
    /**
     * Go forward
     */
    forward() {
        window.history.forward();
    },
    
    /**
     * Get all routes
     */
    getAllRoutes() {
        return Array.from(this.routes.values());
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Router;
}

