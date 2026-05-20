// Package circuitbreaker provides resilient service communication patterns
package circuitbreaker

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/sony/gobreaker"
)

// State represents the circuit breaker state
type State string

const (
	StateClosed   State = "closed"
	StateOpen     State = "open"
	StateHalfOpen State = "half_open"
)

// Config holds circuit breaker configuration
type Config struct {
	Name                   string
	MaxRequests            uint32
	Interval               time.Duration
	Timeout                time.Duration
	ReadyToTrip            func(counts gobreaker.Counts) bool
	OnStateChange          func(name string, from gobreaker.State, to gobreaker.State)
	IsSuccessful           func(err error) bool
}

// DefaultConfig returns default circuit breaker configuration
func DefaultConfig(name string) Config {
	return Config{
		Name:        name,
		MaxRequests: 3,
		Interval:    60 * time.Second,
		Timeout:     30 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
			return counts.Requests >= 5 && failureRatio >= 0.5
		},
	}
}

// Breaker wraps gobreaker.CircuitBreaker with additional functionality
type Breaker struct {
	cb       *gobreaker.CircuitBreaker
	name     string
	mu       sync.RWMutex
	stats    *Stats
}

// Stats holds circuit breaker statistics
type Stats struct {
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	RejectedRequests   int64
	StateChanges       int
	LastStateChange    time.Time
	CurrentState       State
}

// NewBreaker creates a new circuit breaker
func NewBreaker(config Config) *Breaker {
	b := &Breaker{
		name:  config.Name,
		stats: &Stats{CurrentState: StateClosed},
	}

	settings := gobreaker.Settings{
		Name:        config.Name,
		MaxRequests: config.MaxRequests,
		Interval:    config.Interval,
		Timeout:     config.Timeout,
		ReadyToTrip: config.ReadyToTrip,
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			b.mu.Lock()
			b.stats.StateChanges++
			b.stats.LastStateChange = time.Now()
			b.stats.CurrentState = stateToString(to)
			b.mu.Unlock()

			log.Info().
				Str("breaker", name).
				Str("from", stateToString(from)).
				Str("to", stateToString(to)).
				Msg("Circuit breaker state changed")

			if config.OnStateChange != nil {
				config.OnStateChange(name, from, to)
			}
		},
		IsSuccessful: config.IsSuccessful,
	}

	b.cb = gobreaker.NewCircuitBreaker(settings)
	return b
}

// Execute runs a function through the circuit breaker
func (b *Breaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
	b.mu.Lock()
	b.stats.TotalRequests++
	b.mu.Unlock()

	result, err := b.cb.Execute(fn)

	b.mu.Lock()
	if err != nil {
		if errors.Is(err, gobreaker.ErrOpenState) || errors.Is(err, gobreaker.ErrTooManyRequests) {
			b.stats.RejectedRequests++
		} else {
			b.stats.FailedRequests++
		}
	} else {
		b.stats.SuccessfulRequests++
	}
	b.mu.Unlock()

	return result, err
}

// ExecuteWithContext runs a function with context through the circuit breaker
func (b *Breaker) ExecuteWithContext(ctx context.Context, fn func(context.Context) (interface{}, error)) (interface{}, error) {
	b.mu.Lock()
	b.stats.TotalRequests++
	b.mu.Unlock()

	result, err := b.cb.Execute(func() (interface{}, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			return fn(ctx)
		}
	})

	b.mu.Lock()
	if err != nil {
		if errors.Is(err, gobreaker.ErrOpenState) || errors.Is(err, gobreaker.ErrTooManyRequests) {
			b.stats.RejectedRequests++
		} else {
			b.stats.FailedRequests++
		}
	} else {
		b.stats.SuccessfulRequests++
	}
	b.mu.Unlock()

	return result, err
}

// State returns the current state of the circuit breaker
func (b *Breaker) State() State {
	return stateToString(b.cb.State())
}

// GetStats returns circuit breaker statistics
func (b *Breaker) GetStats() Stats {
	b.mu.RLock()
	defer b.mu.RUnlock()
	stats := *b.stats
	stats.CurrentState = b.State()
	return stats
}

// Reset resets the circuit breaker
func (b *Breaker) Reset() {
	b.mu.Lock()
	b.stats = &Stats{CurrentState: StateClosed}
	b.mu.Unlock()
}

func stateToString(s gobreaker.State) State {
	switch s {
	case gobreaker.StateClosed:
		return StateClosed
	case gobreaker.StateOpen:
		return StateOpen
	case gobreaker.StateHalfOpen:
		return StateHalfOpen
	default:
		return StateClosed
	}
}

// Registry manages multiple circuit breakers
type Registry struct {
	breakers map[string]*Breaker
	mu       sync.RWMutex
}

// NewRegistry creates a new circuit breaker registry
func NewRegistry() *Registry {
	return &Registry{
		breakers: make(map[string]*Breaker),
	}
}

// Get returns a circuit breaker by name, creating it if it doesn't exist
func (r *Registry) Get(name string, config ...Config) *Breaker {
	r.mu.RLock()
	breaker, exists := r.breakers[name]
	r.mu.RUnlock()

	if exists {
		return breaker
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if breaker, exists = r.breakers[name]; exists {
		return breaker
	}

	var cfg Config
	if len(config) > 0 {
		cfg = config[0]
	} else {
		cfg = DefaultConfig(name)
	}

	breaker = NewBreaker(cfg)
	r.breakers[name] = breaker

	log.Info().Str("name", name).Msg("Created circuit breaker")

	return breaker
}

// GetStats returns stats for all circuit breakers
func (r *Registry) GetStats() map[string]Stats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := make(map[string]Stats)
	for name, breaker := range r.breakers {
		stats[name] = breaker.GetStats()
	}
	return stats
}

// ResetAll resets all circuit breakers
func (r *Registry) ResetAll() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, breaker := range r.breakers {
		breaker.Reset()
	}
}

// Global registry instance
var globalRegistry = NewRegistry()

// GetBreaker returns a circuit breaker from the global registry
func GetBreaker(name string, config ...Config) *Breaker {
	return globalRegistry.Get(name, config...)
}

// GetAllStats returns stats for all circuit breakers in the global registry
func GetAllStats() map[string]Stats {
	return globalRegistry.GetStats()
}




