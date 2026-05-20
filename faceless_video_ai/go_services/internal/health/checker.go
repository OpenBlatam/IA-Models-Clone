// Package health provides health checking functionality
package health

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

// Status represents health status
type Status string

const (
	StatusHealthy   Status = "healthy"
	StatusDegraded  Status = "degraded"
	StatusUnhealthy Status = "unhealthy"
	StatusUnknown   Status = "unknown"
)

// CheckResult represents the result of a health check
type CheckResult struct {
	Name        string                 `json:"name"`
	Status      Status                 `json:"status"`
	Critical    bool                   `json:"critical"`
	Message     string                 `json:"message,omitempty"`
	Duration    time.Duration          `json:"duration"`
	LastChecked time.Time              `json:"last_checked"`
	Details     map[string]interface{} `json:"details,omitempty"`
}

// OverallHealth represents the overall system health
type OverallHealth struct {
	Status    Status        `json:"status"`
	Timestamp time.Time     `json:"timestamp"`
	Checks    []CheckResult `json:"checks"`
}

// CheckFunc is a function that performs a health check
type CheckFunc func(ctx context.Context) CheckResult

// Check represents a health check
type Check struct {
	Name     string
	Func     CheckFunc
	Critical bool
	Timeout  time.Duration
}

// Checker manages health checks
type Checker struct {
	checks        []Check
	results       map[string]CheckResult
	checkInterval time.Duration
	mu            sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
}

// NewChecker creates a new health checker
func NewChecker(checkInterval time.Duration) *Checker {
	ctx, cancel := context.WithCancel(context.Background())
	c := &Checker{
		checks:        make([]Check, 0),
		results:       make(map[string]CheckResult),
		checkInterval: checkInterval,
		ctx:           ctx,
		cancel:        cancel,
	}

	c.registerDefaultChecks()
	return c
}

func (c *Checker) registerDefaultChecks() {
	c.RegisterCheck(Check{
		Name:     "system",
		Critical: false,
		Timeout:  5 * time.Second,
		Func: func(ctx context.Context) CheckResult {
			return CheckResult{
				Name:    "system",
				Status:  StatusHealthy,
				Message: "System operational",
			}
		},
	})
}

// RegisterCheck registers a health check
func (c *Checker) RegisterCheck(check Check) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if check.Timeout == 0 {
		check.Timeout = 10 * time.Second
	}

	c.checks = append(c.checks, check)
	log.Info().Str("name", check.Name).Bool("critical", check.Critical).Msg("Health check registered")
}

// Start starts periodic health checking
func (c *Checker) Start() {
	go c.runChecks()
	go c.periodicCheck()
}

func (c *Checker) periodicCheck() {
	ticker := time.NewTicker(c.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.runChecks()
		case <-c.ctx.Done():
			return
		}
	}
}

func (c *Checker) runChecks() {
	c.mu.RLock()
	checks := make([]Check, len(c.checks))
	copy(checks, c.checks)
	c.mu.RUnlock()

	var wg sync.WaitGroup
	results := make(chan CheckResult, len(checks))

	for _, check := range checks {
		wg.Add(1)
		go func(ch Check) {
			defer wg.Done()
			result := c.runSingleCheck(ch)
			results <- result
		}(check)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	c.mu.Lock()
	for result := range results {
		c.results[result.Name] = result
	}
	c.mu.Unlock()
}

func (c *Checker) runSingleCheck(check Check) CheckResult {
	ctx, cancel := context.WithTimeout(c.ctx, check.Timeout)
	defer cancel()

	start := time.Now()

	resultChan := make(chan CheckResult, 1)
	go func() {
		result := check.Func(ctx)
		result.Name = check.Name
		result.Critical = check.Critical
		result.Duration = time.Since(start)
		result.LastChecked = time.Now()
		resultChan <- result
	}()

	select {
	case result := <-resultChan:
		return result
	case <-ctx.Done():
		return CheckResult{
			Name:        check.Name,
			Status:      StatusUnhealthy,
			Critical:    check.Critical,
			Message:     "Health check timed out",
			Duration:    time.Since(start),
			LastChecked: time.Now(),
		}
	}
}

// GetHealth returns the overall health status
func (c *Checker) GetHealth() OverallHealth {
	c.mu.RLock()
	defer c.mu.RUnlock()

	checks := make([]CheckResult, 0, len(c.results))
	overallStatus := StatusHealthy

	for _, result := range c.results {
		checks = append(checks, result)

		if result.Status != StatusHealthy {
			if result.Critical {
				overallStatus = StatusUnhealthy
			} else if overallStatus == StatusHealthy {
				overallStatus = StatusDegraded
			}
		}
	}

	return OverallHealth{
		Status:    overallStatus,
		Timestamp: time.Now(),
		Checks:    checks,
	}
}

// GetStatus returns the overall status
func (c *Checker) GetStatus() Status {
	return c.GetHealth().Status
}

// IsHealthy returns true if the system is healthy
func (c *Checker) IsHealthy() bool {
	return c.GetStatus() == StatusHealthy
}

// Stop stops the health checker
func (c *Checker) Stop() {
	c.cancel()
}

// Handler returns an HTTP handler for health checks
func (c *Checker) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		health := c.GetHealth()

		w.Header().Set("Content-Type", "application/json")

		switch health.Status {
		case StatusHealthy:
			w.WriteHeader(http.StatusOK)
		case StatusDegraded:
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusServiceUnavailable)
		}

		json.NewEncoder(w).Encode(health)
	}
}

// LivenessHandler returns a simple liveness probe handler
func (c *Checker) LivenessHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "alive"})
	}
}

// ReadinessHandler returns a readiness probe handler
func (c *Checker) ReadinessHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if c.IsHealthy() {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "ready"})
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			json.NewEncoder(w).Encode(map[string]string{"status": "not_ready"})
		}
	}
}

// Common health check functions

// HTTPCheck creates an HTTP health check
func HTTPCheck(name, url string, timeout time.Duration) Check {
	return Check{
		Name:     name,
		Timeout:  timeout,
		Critical: false,
		Func: func(ctx context.Context) CheckResult {
			client := &http.Client{Timeout: timeout}
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			if err != nil {
				return CheckResult{
					Status:  StatusUnhealthy,
					Message: "Failed to create request: " + err.Error(),
				}
			}

			resp, err := client.Do(req)
			if err != nil {
				return CheckResult{
					Status:  StatusUnhealthy,
					Message: "Request failed: " + err.Error(),
				}
			}
			defer resp.Body.Close()

			if resp.StatusCode >= 400 {
				return CheckResult{
					Status:  StatusUnhealthy,
					Message: "Unhealthy response: " + resp.Status,
				}
			}

			return CheckResult{
				Status:  StatusHealthy,
				Message: "OK",
			}
		},
	}
}

// TCPCheck creates a TCP connectivity health check
func TCPCheck(name, address string, timeout time.Duration) Check {
	return Check{
		Name:     name,
		Timeout:  timeout,
		Critical: true,
		Func: func(ctx context.Context) CheckResult {
			import "net"
			
			conn, err := net.DialTimeout("tcp", address, timeout)
			if err != nil {
				return CheckResult{
					Status:  StatusUnhealthy,
					Message: "Connection failed: " + err.Error(),
				}
			}
			conn.Close()

			return CheckResult{
				Status:  StatusHealthy,
				Message: "Connected",
			}
		},
	}
}




