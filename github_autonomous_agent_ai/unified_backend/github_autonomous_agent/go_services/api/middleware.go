package api

import (
	"net/http"
	"time"

	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/metrics"
	"github.com/rs/zerolog"
)

// Middleware wraps HTTP handlers with common functionality
type Middleware struct {
	logger zerolog.Logger
}

// NewMiddleware creates a new middleware instance
func NewMiddleware(logger zerolog.Logger) *Middleware {
	return &Middleware{
		logger: logger,
	}
}

// LoggingMiddleware logs HTTP requests
func (m *Middleware) LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Create response writer wrapper to capture status code
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(rw, r)
		
		duration := time.Since(start)
		
		m.logger.Info().
			Str("method", r.Method).
			Str("path", r.URL.Path).
			Int("status", rw.statusCode).
			Dur("duration", duration).
			Msg("HTTP request")
		
		// Record metrics
		metrics.HTTPRequestsTotal.WithLabelValues(
			r.Method,
			r.URL.Path,
			http.StatusText(rw.statusCode),
		).Inc()
		
		metrics.HTTPRequestDuration.WithLabelValues(
			r.Method,
			r.URL.Path,
		).Observe(duration.Seconds())
	})
}

// RecoveryMiddleware recovers from panics
func (m *Middleware) RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				m.logger.Error().
					Interface("error", err).
					Str("method", r.Method).
					Str("path", r.URL.Path).
					Msg("Panic recovered")
				
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		}()
		
		next.ServeHTTP(w, r)
	})
}

// CORSMiddleware handles CORS headers
func (m *Middleware) CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		next.ServeHTTP(w, r)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}












