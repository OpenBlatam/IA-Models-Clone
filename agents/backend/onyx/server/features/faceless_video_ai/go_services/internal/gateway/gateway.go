// Package gateway provides an API Gateway implementation with routing,
// middleware, and request/response transformation capabilities.
package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Route represents a gateway route configuration.
type Route struct {
	Path        string            `json:"path"`
	Method      string            `json:"method"`
	Backend     string            `json:"backend"`
	Timeout     time.Duration     `json:"timeout"`
	RateLimit   int               `json:"rate_limit"`
	Auth        bool              `json:"auth"`
	Headers     map[string]string `json:"headers"`
	Middlewares []string          `json:"middlewares"`
	Transform   *TransformConfig  `json:"transform,omitempty"`
}

// TransformConfig defines request/response transformation rules.
type TransformConfig struct {
	RequestHeaders  map[string]string `json:"request_headers,omitempty"`
	ResponseHeaders map[string]string `json:"response_headers,omitempty"`
	BodyTransform   string            `json:"body_transform,omitempty"`
}

// Middleware represents a gateway middleware function.
type Middleware func(http.Handler) http.Handler

// Gateway is the main API Gateway struct.
type Gateway struct {
	mu          sync.RWMutex
	routes      map[string]*Route
	middlewares map[string]Middleware
	proxies     map[string]*httputil.ReverseProxy
	config      *Config
	metrics     *Metrics
}

// Config holds gateway configuration.
type Config struct {
	ListenAddr     string        `json:"listen_addr"`
	ReadTimeout    time.Duration `json:"read_timeout"`
	WriteTimeout   time.Duration `json:"write_timeout"`
	MaxHeaderBytes int           `json:"max_header_bytes"`
	TLSCert        string        `json:"tls_cert,omitempty"`
	TLSKey         string        `json:"tls_key,omitempty"`
	CORSOrigins    []string      `json:"cors_origins"`
}

// Metrics tracks gateway performance metrics.
type Metrics struct {
	mu             sync.RWMutex
	TotalRequests  int64
	SuccessCount   int64
	ErrorCount     int64
	AvgLatencyMs   float64
	RouteMetrics   map[string]*RouteMetrics
	lastResetTime  time.Time
}

// RouteMetrics tracks per-route metrics.
type RouteMetrics struct {
	Requests     int64
	Errors       int64
	AvgLatencyMs float64
	P95LatencyMs float64
	P99LatencyMs float64
	latencies    []float64
}

// New creates a new Gateway instance.
func New(config *Config) *Gateway {
	return &Gateway{
		routes:      make(map[string]*Route),
		middlewares: make(map[string]Middleware),
		proxies:     make(map[string]*httputil.ReverseProxy),
		config:      config,
		metrics: &Metrics{
			RouteMetrics:  make(map[string]*RouteMetrics),
			lastResetTime: time.Now(),
		},
	}
}

// RegisterRoute adds a new route to the gateway.
func (g *Gateway) RegisterRoute(route *Route) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	key := routeKey(route.Method, route.Path)
	g.routes[key] = route

	// Create reverse proxy for the backend
	if route.Backend != "" {
		backendURL, err := url.Parse(route.Backend)
		if err != nil {
			return fmt.Errorf("invalid backend URL: %w", err)
		}
		g.proxies[key] = httputil.NewSingleHostReverseProxy(backendURL)
	}

	// Initialize metrics for the route
	g.metrics.RouteMetrics[key] = &RouteMetrics{
		latencies: make([]float64, 0, 1000),
	}

	return nil
}

// RegisterMiddleware adds a named middleware.
func (g *Gateway) RegisterMiddleware(name string, mw Middleware) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.middlewares[name] = mw
}

// Handler returns the main HTTP handler for the gateway.
func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/health", g.healthHandler)

	// Metrics endpoint
	mux.HandleFunc("/metrics", g.metricsHandler)

	// Routes endpoint
	mux.HandleFunc("/routes", g.routesHandler)

	// Catch-all for gateway routing
	mux.HandleFunc("/", g.routeHandler)

	// Apply global middlewares
	var handler http.Handler = mux
	handler = g.corsMiddleware(handler)
	handler = g.loggingMiddleware(handler)
	handler = g.recoveryMiddleware(handler)
	handler = g.requestIDMiddleware(handler)

	return handler
}

// Start starts the gateway server.
func (g *Gateway) Start() error {
	server := &http.Server{
		Addr:           g.config.ListenAddr,
		Handler:        g.Handler(),
		ReadTimeout:    g.config.ReadTimeout,
		WriteTimeout:   g.config.WriteTimeout,
		MaxHeaderBytes: g.config.MaxHeaderBytes,
	}

	if g.config.TLSCert != "" && g.config.TLSKey != "" {
		return server.ListenAndServeTLS(g.config.TLSCert, g.config.TLSKey)
	}
	return server.ListenAndServe()
}

// routeHandler handles incoming requests and routes them to backends.
func (g *Gateway) routeHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	key := routeKey(r.Method, r.URL.Path)

	g.mu.RLock()
	route, exists := g.routes[key]
	proxy := g.proxies[key]
	g.mu.RUnlock()

	if !exists {
		// Try wildcard matching
		route, proxy = g.findWildcardRoute(r.Method, r.URL.Path)
		if route == nil {
			http.Error(w, "Not Found", http.StatusNotFound)
			g.recordMetrics(key, time.Since(startTime), true)
			return
		}
	}

	// Apply route-specific middlewares
	var handler http.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if proxy != nil {
			// Apply request transformation
			if route.Transform != nil {
				g.transformRequest(r, route.Transform)
			}

			// Add custom headers
			for k, v := range route.Headers {
				r.Header.Set(k, v)
			}

			// Forward to backend
			proxy.ServeHTTP(w, r)
		} else {
			http.Error(w, "No backend configured", http.StatusBadGateway)
		}
	})

	// Apply route middlewares in reverse order
	for i := len(route.Middlewares) - 1; i >= 0; i-- {
		if mw, ok := g.middlewares[route.Middlewares[i]]; ok {
			handler = mw(handler)
		}
	}

	// Set timeout context
	ctx, cancel := context.WithTimeout(r.Context(), route.Timeout)
	defer cancel()

	handler.ServeHTTP(w, r.WithContext(ctx))
	g.recordMetrics(key, time.Since(startTime), false)
}

// findWildcardRoute finds a matching wildcard route.
func (g *Gateway) findWildcardRoute(method, path string) (*Route, *httputil.ReverseProxy) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	for key, route := range g.routes {
		if matchesPattern(key, routeKey(method, path)) {
			return route, g.proxies[key]
		}
	}
	return nil, nil
}

// matchesPattern checks if a path matches a pattern with wildcards.
func matchesPattern(pattern, path string) bool {
	patternParts := strings.Split(pattern, "/")
	pathParts := strings.Split(path, "/")

	if len(patternParts) != len(pathParts) {
		return false
	}

	for i, part := range patternParts {
		if part == "*" || part == pathParts[i] {
			continue
		}
		if strings.HasPrefix(part, ":") {
			continue // Path parameter
		}
		return false
	}
	return true
}

// transformRequest applies transformations to the request.
func (g *Gateway) transformRequest(r *http.Request, config *TransformConfig) {
	if config.RequestHeaders != nil {
		for k, v := range config.RequestHeaders {
			r.Header.Set(k, v)
		}
	}
}

// recordMetrics records request metrics.
func (g *Gateway) recordMetrics(routeKey string, duration time.Duration, isError bool) {
	g.metrics.mu.Lock()
	defer g.metrics.mu.Unlock()

	g.metrics.TotalRequests++
	latencyMs := float64(duration.Milliseconds())

	if isError {
		g.metrics.ErrorCount++
	} else {
		g.metrics.SuccessCount++
	}

	// Update average latency
	g.metrics.AvgLatencyMs = (g.metrics.AvgLatencyMs*float64(g.metrics.TotalRequests-1) + latencyMs) / float64(g.metrics.TotalRequests)

	// Update route-specific metrics
	if rm, ok := g.metrics.RouteMetrics[routeKey]; ok {
		rm.Requests++
		if isError {
			rm.Errors++
		}
		rm.latencies = append(rm.latencies, latencyMs)
		rm.AvgLatencyMs = (rm.AvgLatencyMs*float64(rm.Requests-1) + latencyMs) / float64(rm.Requests)
	}
}

// healthHandler handles health check requests.
func (g *Gateway) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"uptime": time.Since(g.metrics.lastResetTime).String(),
	})
}

// metricsHandler returns gateway metrics.
func (g *Gateway) metricsHandler(w http.ResponseWriter, r *http.Request) {
	g.metrics.mu.RLock()
	defer g.metrics.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(g.metrics)
}

// routesHandler returns registered routes.
func (g *Gateway) routesHandler(w http.ResponseWriter, r *http.Request) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	routes := make([]*Route, 0, len(g.routes))
	for _, route := range g.routes {
		routes = append(routes, route)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(routes)
}

// Middleware implementations

func (g *Gateway) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = uuid.New().String()
		}
		w.Header().Set("X-Request-ID", requestID)
		r.Header.Set("X-Request-ID", requestID)
		next.ServeHTTP(w, r)
	})
}

func (g *Gateway) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(wrapped, r)
		// Log would go here
		_ = fmt.Sprintf("[%s] %s %s %d %v",
			r.Header.Get("X-Request-ID"),
			r.Method,
			r.URL.Path,
			wrapped.statusCode,
			time.Since(start))
	})
}

func (g *Gateway) recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (g *Gateway) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		allowed := false

		for _, o := range g.config.CORSOrigins {
			if o == "*" || o == origin {
				allowed = true
				break
			}
		}

		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
			w.Header().Set("Access-Control-Max-Age", "86400")
		}

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// routeKey generates a unique key for a route.
func routeKey(method, path string) string {
	return fmt.Sprintf("%s:%s", strings.ToUpper(method), path)
}

// DefaultConfig returns a default gateway configuration.
func DefaultConfig() *Config {
	return &Config{
		ListenAddr:     ":8080",
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   30 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1MB
		CORSOrigins:    []string{"*"},
	}
}




