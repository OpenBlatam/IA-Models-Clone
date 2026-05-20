package api

import (
	"net/http"
)

// SetupRoutes configures all API routes with middleware
func (h *Handlers) SetupRoutes(mux *http.ServeMux, middleware *Middleware) {
	// Apply middleware
	handler := middleware.RecoveryMiddleware(
		middleware.LoggingMiddleware(
			middleware.CORSMiddleware(
				http.DefaultServeMux,
			),
		),
	)

	// Health check (no middleware needed)
	mux.HandleFunc("/health", h.HealthHandler)
	mux.HandleFunc("/api/v1/health", h.HealthHandler)

	// Git operations
	mux.HandleFunc("/api/v1/git/clone", h.GitCloneHandler)

	// Search
	mux.HandleFunc("/api/v1/search", h.SearchHandler)

	// Cache operations
	mux.HandleFunc("/api/v1/cache", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			h.CacheGetHandler(w, r)
		case http.MethodPost:
			h.CacheSetHandler(w, r)
		case http.MethodDelete:
			h.CacheDeleteHandler(w, r)
		default:
			h.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		}
	})

	// Metrics endpoint
	mux.HandleFunc("/metrics", h.MetricsHandler)
}

// MetricsHandler handles Prometheus metrics
func (h *Handlers) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	// This would expose Prometheus metrics
	// For now, return basic stats
	h.writeSuccess(w, http.StatusOK, map[string]interface{}{
		"note": "Prometheus metrics to be implemented",
	})
}

