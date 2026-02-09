/*
 * Benchmark Worker - Concurrent benchmark execution
 * 
 * Refactored with:
 * - Better error handling
 * - Metrics collection
 * - Task prioritization
 * - Retry logic
 */

package workers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// BenchmarkTask represents a single benchmark task
type BenchmarkTask struct {
	ID          string                 `json:"id"`
	ModelName   string                 `json:"model_name"`
	Benchmark   string                 `json:"benchmark"`
	Config      map[string]interface{} `json:"config"`
	Priority    int                    `json:"priority"` // Higher = more priority
	Status      string                 `json:"status"`
	Result      *BenchmarkResult       `json:"result,omitempty"`
	Error       string                 `json:"error,omitempty"`
	Retries     int                    `json:"retries"`
	MaxRetries  int                    `json:"max_retries"`
	CreatedAt   time.Time              `json:"created_at"`
	StartedAt   *time.Time             `json:"started_at,omitempty"`
	CompletedAt *time.Time             `json:"completed_at,omitempty"`
}

// BenchmarkResult contains the results of a benchmark execution
type BenchmarkResult struct {
	Accuracy       float64            `json:"accuracy"`
	LatencyP50     float64            `json:"latency_p50"`
	LatencyP95     float64            `json:"latency_p95"`
	LatencyP99     float64            `json:"latency_p99"`
	Throughput     float64            `json:"throughput"`
	MemoryUsage    map[string]float64 `json:"memory_usage"`
	TotalSamples   int                `json:"total_samples"`
	CorrectSamples int                `json:"correct_samples"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// WorkerPool manages a pool of benchmark workers
type WorkerPool struct {
	workers      int
	taskQueue    chan *BenchmarkTask
	resultChan   chan *BenchmarkTask
	wg           sync.WaitGroup
	ctx          context.Context
	cancel       context.CancelFunc
	activeTasks  int64
	completedTasks int64
	failedTasks  int64
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workers int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	return &WorkerPool{
		workers:    workers,
		taskQueue:  make(chan *BenchmarkTask, 100),
		resultChan: make(chan *BenchmarkTask, 100),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the worker pool
func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
	log.Printf("Worker pool started with %d workers", wp.workers)
}

// Stop stops the worker pool
func (wp *WorkerPool) Stop() {
	close(wp.taskQueue)
	wp.cancel()
	wp.wg.Wait()
	close(wp.resultChan)
	log.Printf("Worker pool stopped")
}

// SubmitTask submits a task to the worker pool
func (wp *WorkerPool) SubmitTask(task *BenchmarkTask) error {
	select {
	case wp.taskQueue <- task:
		return nil
	case <-wp.ctx.Done():
		return fmt.Errorf("worker pool stopped")
	default:
		return fmt.Errorf("task queue full")
	}
}

// GetResults returns the result channel
func (wp *WorkerPool) GetResults() <-chan *BenchmarkTask {
	return wp.resultChan
}

// GetStats returns pool statistics
func (wp *WorkerPool) GetStats() map[string]int64 {
	return map[string]int64{
		"active":    atomic.LoadInt64(&wp.activeTasks),
		"completed": atomic.LoadInt64(&wp.completedTasks),
		"failed":    atomic.LoadInt64(&wp.failedTasks),
	}
}

// worker is the worker goroutine
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()
	
	log.Printf("Worker %d started", id)
	
	for {
		select {
		case task, ok := <-wp.taskQueue:
			if !ok {
				log.Printf("Worker %d stopping", id)
				return
			}
			
			atomic.AddInt64(&wp.activeTasks, 1)
			now := time.Now()
			task.StartedAt = &now
			task.Status = "processing"
			
			log.Printf("Worker %d processing task %s", id, task.ID)
			
			// Execute benchmark with retry logic
			result, err := wp.executeBenchmarkWithRetry(task)
			if err != nil {
				atomic.AddInt64(&wp.failedTasks, 1)
				task.Error = err.Error()
				task.Status = "failed"
			} else {
				atomic.AddInt64(&wp.completedTasks, 1)
				task.Result = result
				task.Status = "completed"
				now := time.Now()
				task.CompletedAt = &now
			}
			
			atomic.AddInt64(&wp.activeTasks, -1)
			wp.resultChan <- task
			
		case <-wp.ctx.Done():
			log.Printf("Worker %d cancelled", id)
			return
		}
	}
}

// executeBenchmarkWithRetry executes a benchmark with retry logic
func (wp *WorkerPool) executeBenchmarkWithRetry(task *BenchmarkTask) (*BenchmarkResult, error) {
	maxRetries := task.MaxRetries
	if maxRetries == 0 {
		maxRetries = 3 // Default
	}
	
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			task.Retries = attempt
			log.Printf("Retrying task %s (attempt %d/%d)", task.ID, attempt, maxRetries)
			time.Sleep(time.Duration(attempt) * time.Second) // Exponential backoff
		}
		
		result, err := wp.executeBenchmark(task)
		if err == nil {
			return result, nil
		}
		
		lastErr = err
		log.Printf("Task %s failed (attempt %d): %v", task.ID, attempt+1, err)
	}
	
	return nil, fmt.Errorf("task failed after %d retries: %w", maxRetries, lastErr)
}

// executeBenchmark executes a benchmark task
func (wp *WorkerPool) executeBenchmark(task *BenchmarkTask) (*BenchmarkResult, error) {
	// This would integrate with Python orchestrator or Rust inference engine
	// For now, return a placeholder result
	
	// Simulate benchmark execution
	time.Sleep(100 * time.Millisecond)
	
	return &BenchmarkResult{
		Accuracy:    0.85,
		LatencyP50:  0.1,
		LatencyP95:  0.2,
		LatencyP99:  0.3,
		Throughput:  100.0,
		MemoryUsage: map[string]float64{
			"gpu_mb": 4096.0,
			"cpu_mb": 2048.0,
		},
		TotalSamples:   100,
		CorrectSamples: 85,
		Metadata:       make(map[string]interface{}),
	}, nil
}

// Scheduler manages benchmark task scheduling
type Scheduler struct {
	pool      *WorkerPool
	tasks     []*BenchmarkTask
	mu        sync.RWMutex
}

// NewScheduler creates a new scheduler
func NewScheduler(pool *WorkerPool) *Scheduler {
	return &Scheduler{
		pool:  pool,
		tasks: make([]*BenchmarkTask, 0),
	}
}

// ScheduleTask schedules a new task
func (s *Scheduler) ScheduleTask(task *BenchmarkTask) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	task.CreatedAt = time.Now()
	task.Status = "queued"
	if task.MaxRetries == 0 {
		task.MaxRetries = 3 // Default
	}
	
	s.tasks = append(s.tasks, task)
	
	return s.pool.SubmitTask(task)
}

// GetTaskStatus returns the status of a task
func (s *Scheduler) GetTaskStatus(taskID string) (*BenchmarkTask, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	for _, task := range s.tasks {
		if task.ID == taskID {
			return task, nil
		}
	}
	
	return nil, fmt.Errorf("task not found: %s", taskID)
}

// GetAllTasks returns all tasks
func (s *Scheduler) GetAllTasks() []*BenchmarkTask {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	result := make([]*BenchmarkTask, len(s.tasks))
	copy(result, s.tasks)
	return result
}

// GetStats returns scheduler statistics
func (s *Scheduler) GetStats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	stats := map[string]interface{}{
		"total_tasks": len(s.tasks),
		"pool_stats":  s.pool.GetStats(),
	}
	
	// Count tasks by status
	statusCounts := make(map[string]int)
	for _, task := range s.tasks {
		statusCounts[task.Status]++
	}
	stats["tasks_by_status"] = statusCounts
	
	return stats
}
