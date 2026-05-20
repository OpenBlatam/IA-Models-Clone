// Package ratelimit provides high-performance rate limiting
package ratelimit

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// Limiter provides rate limiting functionality
type Limiter struct {
	limiters map[string]*rateLimiterEntry
	mu       sync.RWMutex
	cleanup  time.Duration
}

type rateLimiterEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// Config holds rate limiter configuration
type Config struct {
	Rate       float64
	Burst      int
	CleanupAge time.Duration
}

// DefaultConfig returns default rate limiter configuration
func DefaultConfig() Config {
	return Config{
		Rate:       100,
		Burst:      200,
		CleanupAge: 1 * time.Hour,
	}
}

// Result contains rate limit check result
type Result struct {
	Allowed   bool  `json:"allowed"`
	Remaining int   `json:"remaining"`
	ResetAt   int64 `json:"reset_at"`
	Limit     int   `json:"limit"`
}

// NewLimiter creates a new rate limiter
func NewLimiter(config Config) *Limiter {
	l := &Limiter{
		limiters: make(map[string]*rateLimiterEntry),
		cleanup:  config.CleanupAge,
	}

	go l.cleanupLoop()
	return l
}

// Allow checks if a request is allowed
func (l *Limiter) Allow(key string, r float64, b int) Result {
	l.mu.Lock()
	entry, exists := l.limiters[key]
	if !exists {
		entry = &rateLimiterEntry{
			limiter:  rate.NewLimiter(rate.Limit(r), b),
			lastSeen: time.Now(),
		}
		l.limiters[key] = entry
	}
	entry.lastSeen = time.Now()
	l.mu.Unlock()

	allowed := entry.limiter.Allow()
	reservation := entry.limiter.Reserve()
	defer reservation.Cancel()

	tokens := int(entry.limiter.Tokens())
	resetAt := time.Now().Add(reservation.Delay()).Unix()

	return Result{
		Allowed:   allowed,
		Remaining: tokens,
		ResetAt:   resetAt,
		Limit:     b,
	}
}

// AllowN checks if n requests are allowed
func (l *Limiter) AllowN(key string, r float64, b int, n int) Result {
	l.mu.Lock()
	entry, exists := l.limiters[key]
	if !exists {
		entry = &rateLimiterEntry{
			limiter:  rate.NewLimiter(rate.Limit(r), b),
			lastSeen: time.Now(),
		}
		l.limiters[key] = entry
	}
	entry.lastSeen = time.Now()
	l.mu.Unlock()

	allowed := entry.limiter.AllowN(time.Now(), n)
	tokens := int(entry.limiter.Tokens())

	return Result{
		Allowed:   allowed,
		Remaining: tokens,
		ResetAt:   time.Now().Add(time.Duration(float64(n)/r) * time.Second).Unix(),
		Limit:     b,
	}
}

// Wait blocks until the rate limit allows or context is cancelled
func (l *Limiter) Wait(key string, r float64, b int) error {
	l.mu.Lock()
	entry, exists := l.limiters[key]
	if !exists {
		entry = &rateLimiterEntry{
			limiter:  rate.NewLimiter(rate.Limit(r), b),
			lastSeen: time.Now(),
		}
		l.limiters[key] = entry
	}
	entry.lastSeen = time.Now()
	l.mu.Unlock()

	return entry.limiter.Wait(nil)
}

// Reset resets the rate limiter for a key
func (l *Limiter) Reset(key string) {
	l.mu.Lock()
	delete(l.limiters, key)
	l.mu.Unlock()
}

// ResetAll resets all rate limiters
func (l *Limiter) ResetAll() {
	l.mu.Lock()
	l.limiters = make(map[string]*rateLimiterEntry)
	l.mu.Unlock()
}

// GetStats returns limiter statistics
func (l *Limiter) GetStats() map[string]interface{} {
	l.mu.RLock()
	defer l.mu.RUnlock()

	return map[string]interface{}{
		"active_limiters": len(l.limiters),
	}
}

func (l *Limiter) cleanupLoop() {
	ticker := time.NewTicker(l.cleanup / 2)
	defer ticker.Stop()

	for range ticker.C {
		l.cleanup_old()
	}
}

func (l *Limiter) cleanup_old() {
	l.mu.Lock()
	defer l.mu.Unlock()

	threshold := time.Now().Add(-l.cleanup)
	for key, entry := range l.limiters {
		if entry.lastSeen.Before(threshold) {
			delete(l.limiters, key)
		}
	}
}

// SlidingWindowLimiter implements sliding window rate limiting
type SlidingWindowLimiter struct {
	windows map[string]*slidingWindow
	mu      sync.RWMutex
}

type slidingWindow struct {
	requests []time.Time
	mu       sync.Mutex
}

// NewSlidingWindowLimiter creates a new sliding window limiter
func NewSlidingWindowLimiter() *SlidingWindowLimiter {
	return &SlidingWindowLimiter{
		windows: make(map[string]*slidingWindow),
	}
}

// Allow checks if a request is allowed within the sliding window
func (l *SlidingWindowLimiter) Allow(key string, maxRequests int, windowSize time.Duration) Result {
	l.mu.Lock()
	window, exists := l.windows[key]
	if !exists {
		window = &slidingWindow{
			requests: make([]time.Time, 0),
		}
		l.windows[key] = window
	}
	l.mu.Unlock()

	window.mu.Lock()
	defer window.mu.Unlock()

	now := time.Now()
	threshold := now.Add(-windowSize)

	validRequests := make([]time.Time, 0)
	for _, t := range window.requests {
		if t.After(threshold) {
			validRequests = append(validRequests, t)
		}
	}
	window.requests = validRequests

	if len(window.requests) >= maxRequests {
		resetAt := window.requests[0].Add(windowSize).Unix()
		return Result{
			Allowed:   false,
			Remaining: 0,
			ResetAt:   resetAt,
			Limit:     maxRequests,
		}
	}

	window.requests = append(window.requests, now)

	var resetAt int64
	if len(window.requests) > 0 {
		resetAt = window.requests[0].Add(windowSize).Unix()
	}

	return Result{
		Allowed:   true,
		Remaining: maxRequests - len(window.requests),
		ResetAt:   resetAt,
		Limit:     maxRequests,
	}
}

// Reset resets the sliding window for a key
func (l *SlidingWindowLimiter) Reset(key string) {
	l.mu.Lock()
	delete(l.windows, key)
	l.mu.Unlock()
}

// MultiLimiter combines multiple rate limiters
type MultiLimiter struct {
	limiters map[string]*Limiter
	configs  map[string]LimitConfig
	mu       sync.RWMutex
}

// LimitConfig holds configuration for a named limit
type LimitConfig struct {
	Rate  float64
	Burst int
}

// NewMultiLimiter creates a new multi-limiter
func NewMultiLimiter() *MultiLimiter {
	return &MultiLimiter{
		limiters: make(map[string]*Limiter),
		configs:  make(map[string]LimitConfig),
	}
}

// SetLimit sets a named limit configuration
func (m *MultiLimiter) SetLimit(name string, config LimitConfig) {
	m.mu.Lock()
	m.configs[name] = config
	m.limiters[name] = NewLimiter(Config{
		Rate:       config.Rate,
		Burst:      config.Burst,
		CleanupAge: 1 * time.Hour,
	})
	m.mu.Unlock()
}

// Allow checks if a request is allowed for a named limit
func (m *MultiLimiter) Allow(limitName, key string) Result {
	m.mu.RLock()
	limiter, exists := m.limiters[limitName]
	config := m.configs[limitName]
	m.mu.RUnlock()

	if !exists {
		return Result{Allowed: true, Remaining: -1, Limit: -1}
	}

	return limiter.Allow(key, config.Rate, config.Burst)
}




