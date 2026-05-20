// Package gateway provides middleware implementations for the API Gateway.
package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// AuthMiddleware provides JWT authentication middleware.
type AuthMiddleware struct {
	secretKey     []byte
	excludePaths  []string
	tokenHeader   string
}

// NewAuthMiddleware creates a new auth middleware.
func NewAuthMiddleware(secretKey string, excludePaths []string) *AuthMiddleware {
	return &AuthMiddleware{
		secretKey:    []byte(secretKey),
		excludePaths: excludePaths,
		tokenHeader:  "Authorization",
	}
}

// Handler returns the middleware handler.
func (am *AuthMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if path is excluded
		for _, path := range am.excludePaths {
			if r.URL.Path == path {
				next.ServeHTTP(w, r)
				return
			}
		}

		token := r.Header.Get(am.tokenHeader)
		if token == "" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Remove "Bearer " prefix if present
		if len(token) > 7 && token[:7] == "Bearer " {
			token = token[7:]
		}

		// Validate token (simplified - use proper JWT validation in production)
		claims, err := am.validateToken(token)
		if err != nil {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		// Add claims to context
		ctx := context.WithValue(r.Context(), "user", claims)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (am *AuthMiddleware) validateToken(token string) (map[string]interface{}, error) {
	// Simplified token validation
	// In production, use proper JWT library
	return map[string]interface{}{
		"user_id": "user123",
		"role":    "user",
	}, nil
}

// RateLimitMiddleware provides rate limiting.
type RateLimitMiddleware struct {
	mu           sync.RWMutex
	limits       map[string]*rateLimitEntry
	requestLimit int
	windowSize   time.Duration
	keyFunc      func(*http.Request) string
}

type rateLimitEntry struct {
	count     int
	resetTime time.Time
}

// NewRateLimitMiddleware creates a new rate limit middleware.
func NewRateLimitMiddleware(requestLimit int, windowSize time.Duration) *RateLimitMiddleware {
	rl := &RateLimitMiddleware{
		limits:       make(map[string]*rateLimitEntry),
		requestLimit: requestLimit,
		windowSize:   windowSize,
		keyFunc:      defaultKeyFunc,
	}

	// Start cleanup goroutine
	go rl.cleanup()

	return rl
}

func defaultKeyFunc(r *http.Request) string {
	// Use IP address as default key
	ip := r.Header.Get("X-Forwarded-For")
	if ip == "" {
		ip = r.Header.Get("X-Real-IP")
	}
	if ip == "" {
		ip = r.RemoteAddr
	}
	return ip
}

// Handler returns the middleware handler.
func (rl *RateLimitMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := rl.keyFunc(r)

		rl.mu.Lock()
		entry, exists := rl.limits[key]
		now := time.Now()

		if !exists || now.After(entry.resetTime) {
			rl.limits[key] = &rateLimitEntry{
				count:     1,
				resetTime: now.Add(rl.windowSize),
			}
			rl.mu.Unlock()
			next.ServeHTTP(w, r)
			return
		}

		entry.count++
		if entry.count > rl.requestLimit {
			rl.mu.Unlock()
			w.Header().Set("X-RateLimit-Limit", string(rune(rl.requestLimit)))
			w.Header().Set("X-RateLimit-Remaining", "0")
			w.Header().Set("Retry-After", entry.resetTime.Format(time.RFC1123))
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			return
		}

		remaining := rl.requestLimit - entry.count
		rl.mu.Unlock()

		w.Header().Set("X-RateLimit-Limit", string(rune(rl.requestLimit)))
		w.Header().Set("X-RateLimit-Remaining", string(rune(remaining)))

		next.ServeHTTP(w, r)
	})
}

func (rl *RateLimitMiddleware) cleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		rl.mu.Lock()
		now := time.Now()
		for key, entry := range rl.limits {
			if now.After(entry.resetTime) {
				delete(rl.limits, key)
			}
		}
		rl.mu.Unlock()
	}
}

// CacheMiddleware provides response caching.
type CacheMiddleware struct {
	mu         sync.RWMutex
	cache      map[string]*cacheEntry
	defaultTTL time.Duration
	maxSize    int
}

type cacheEntry struct {
	body       []byte
	headers    http.Header
	statusCode int
	expiresAt  time.Time
}

// NewCacheMiddleware creates a new cache middleware.
func NewCacheMiddleware(defaultTTL time.Duration, maxSize int) *CacheMiddleware {
	cm := &CacheMiddleware{
		cache:      make(map[string]*cacheEntry),
		defaultTTL: defaultTTL,
		maxSize:    maxSize,
	}

	go cm.cleanup()

	return cm
}

// Handler returns the middleware handler.
func (cm *CacheMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only cache GET requests
		if r.Method != http.MethodGet {
			next.ServeHTTP(w, r)
			return
		}

		cacheKey := r.URL.String()

		// Check cache
		cm.mu.RLock()
		entry, exists := cm.cache[cacheKey]
		cm.mu.RUnlock()

		if exists && time.Now().Before(entry.expiresAt) {
			// Serve from cache
			for k, v := range entry.headers {
				w.Header()[k] = v
			}
			w.Header().Set("X-Cache", "HIT")
			w.WriteHeader(entry.statusCode)
			w.Write(entry.body)
			return
		}

		// Capture response
		rec := &responseRecorder{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
			body:           make([]byte, 0),
		}

		next.ServeHTTP(rec, r)

		// Store in cache if successful
		if rec.statusCode == http.StatusOK {
			cm.mu.Lock()
			if len(cm.cache) < cm.maxSize {
				cm.cache[cacheKey] = &cacheEntry{
					body:       rec.body,
					headers:    rec.Header().Clone(),
					statusCode: rec.statusCode,
					expiresAt:  time.Now().Add(cm.defaultTTL),
				}
			}
			cm.mu.Unlock()
		}

		w.Header().Set("X-Cache", "MISS")
	})
}

func (cm *CacheMiddleware) cleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		cm.mu.Lock()
		now := time.Now()
		for key, entry := range cm.cache {
			if now.After(entry.expiresAt) {
				delete(cm.cache, key)
			}
		}
		cm.mu.Unlock()
	}
}

// Invalidate removes an entry from the cache.
func (cm *CacheMiddleware) Invalidate(pattern string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	delete(cm.cache, pattern)
}

// responseRecorder captures the response for caching.
type responseRecorder struct {
	http.ResponseWriter
	statusCode int
	body       []byte
}

func (rec *responseRecorder) WriteHeader(code int) {
	rec.statusCode = code
	rec.ResponseWriter.WriteHeader(code)
}

func (rec *responseRecorder) Write(b []byte) (int, error) {
	rec.body = append(rec.body, b...)
	return rec.ResponseWriter.Write(b)
}

// CompressionMiddleware provides gzip compression.
type CompressionMiddleware struct {
	minSize int
}

// NewCompressionMiddleware creates a new compression middleware.
func NewCompressionMiddleware(minSize int) *CompressionMiddleware {
	return &CompressionMiddleware{minSize: minSize}
}

// TimeoutMiddleware provides request timeout handling.
type TimeoutMiddleware struct {
	timeout time.Duration
}

// NewTimeoutMiddleware creates a new timeout middleware.
func NewTimeoutMiddleware(timeout time.Duration) *TimeoutMiddleware {
	return &TimeoutMiddleware{timeout: timeout}
}

// Handler returns the middleware handler.
func (tm *TimeoutMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), tm.timeout)
		defer cancel()

		done := make(chan struct{})
		go func() {
			next.ServeHTTP(w, r.WithContext(ctx))
			close(done)
		}()

		select {
		case <-done:
			return
		case <-ctx.Done():
			http.Error(w, "Request timeout", http.StatusGatewayTimeout)
		}
	})
}

// MetricsMiddleware collects request metrics.
type MetricsMiddleware struct {
	mu       sync.RWMutex
	counters map[string]int64
	latencies map[string][]float64
}

// NewMetricsMiddleware creates a new metrics middleware.
func NewMetricsMiddleware() *MetricsMiddleware {
	return &MetricsMiddleware{
		counters:  make(map[string]int64),
		latencies: make(map[string][]float64),
	}
}

// Handler returns the middleware handler.
func (mm *MetricsMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)
		path := r.URL.Path
		method := r.Method

		mm.mu.Lock()
		key := method + ":" + path
		mm.counters[key]++
		mm.latencies[key] = append(mm.latencies[key], float64(duration.Milliseconds()))
		mm.mu.Unlock()
	})
}

// GetMetrics returns collected metrics.
func (mm *MetricsMiddleware) GetMetrics() map[string]interface{} {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	return map[string]interface{}{
		"counters":  mm.counters,
		"latencies": mm.latencies,
	}
}

// SecurityHeadersMiddleware adds security headers.
type SecurityHeadersMiddleware struct {
	headers map[string]string
}

// NewSecurityHeadersMiddleware creates a new security headers middleware.
func NewSecurityHeadersMiddleware() *SecurityHeadersMiddleware {
	return &SecurityHeadersMiddleware{
		headers: map[string]string{
			"X-Content-Type-Options":    "nosniff",
			"X-Frame-Options":           "DENY",
			"X-XSS-Protection":          "1; mode=block",
			"Strict-Transport-Security": "max-age=31536000; includeSubDomains",
			"Content-Security-Policy":   "default-src 'self'",
			"Referrer-Policy":           "strict-origin-when-cross-origin",
		},
	}
}

// Handler returns the middleware handler.
func (shm *SecurityHeadersMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for k, v := range shm.headers {
			w.Header().Set(k, v)
		}
		next.ServeHTTP(w, r)
	})
}

// RequestValidationMiddleware validates incoming requests.
type RequestValidationMiddleware struct {
	maxBodySize int64
	contentTypes []string
}

// NewRequestValidationMiddleware creates a new request validation middleware.
func NewRequestValidationMiddleware(maxBodySize int64, contentTypes []string) *RequestValidationMiddleware {
	return &RequestValidationMiddleware{
		maxBodySize:  maxBodySize,
		contentTypes: contentTypes,
	}
}

// Handler returns the middleware handler.
func (rvm *RequestValidationMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check content length
		if r.ContentLength > rvm.maxBodySize {
			http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
			return
		}

		// Check content type for POST/PUT/PATCH
		if r.Method == http.MethodPost || r.Method == http.MethodPut || r.Method == http.MethodPatch {
			contentType := r.Header.Get("Content-Type")
			valid := false
			for _, ct := range rvm.contentTypes {
				if contentType == ct {
					valid = true
					break
				}
			}
			if !valid && len(rvm.contentTypes) > 0 {
				http.Error(w, "Invalid content type", http.StatusUnsupportedMediaType)
				return
			}
		}

		next.ServeHTTP(w, r)
	})
}

// ErrorResponse represents an API error response.
type ErrorResponse struct {
	Error   string `json:"error"`
	Code    string `json:"code"`
	Details string `json:"details,omitempty"`
}

// writeError writes a JSON error response.
func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ErrorResponse{
		Error: message,
		Code:  code,
	})
}




