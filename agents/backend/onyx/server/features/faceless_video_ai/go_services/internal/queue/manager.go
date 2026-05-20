// Package queue provides a high-performance job queue with priority support
package queue

import (
	"container/heap"
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
)

// Priority levels for jobs
type Priority int

const (
	PriorityLow    Priority = 0
	PriorityNormal Priority = 1
	PriorityHigh   Priority = 2
	PriorityUrgent Priority = 3
)

func (p Priority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityUrgent:
		return "urgent"
	default:
		return "unknown"
	}
}

// JobStatus represents the status of a job
type JobStatus string

const (
	StatusPending    JobStatus = "pending"
	StatusProcessing JobStatus = "processing"
	StatusCompleted  JobStatus = "completed"
	StatusFailed     JobStatus = "failed"
	StatusCancelled  JobStatus = "cancelled"
)

// Job represents a job in the queue
type Job struct {
	ID          uuid.UUID       `json:"id"`
	Type        string          `json:"type"`
	Priority    Priority        `json:"priority"`
	Data        json.RawMessage `json:"data"`
	Status      JobStatus       `json:"status"`
	Result      json.RawMessage `json:"result,omitempty"`
	Error       string          `json:"error,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	StartedAt   *time.Time      `json:"started_at,omitempty"`
	CompletedAt *time.Time      `json:"completed_at,omitempty"`
	Retries     int             `json:"retries"`
	MaxRetries  int             `json:"max_retries"`
	index       int
}

// JobHandler is a function that processes a job
type JobHandler func(ctx context.Context, job *Job) (json.RawMessage, error)

// PriorityQueue implements heap.Interface for jobs
type PriorityQueue []*Job

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority > pq[j].Priority
	}
	return pq[i].CreatedAt.Before(pq[j].CreatedAt)
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	job := x.(*Job)
	job.index = n
	*pq = append(*pq, job)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	job := old[n-1]
	old[n-1] = nil
	job.index = -1
	*pq = old[0 : n-1]
	return job
}

// Manager manages the job queue
type Manager struct {
	queue      PriorityQueue
	processing map[uuid.UUID]*Job
	completed  map[uuid.UUID]*Job
	handlers   map[string]JobHandler
	workers    int
	running    bool
	mu         sync.RWMutex
	cond       *sync.Cond
	ctx        context.Context
	cancel     context.CancelFunc
}

// NewManager creates a new queue manager
func NewManager(workers int) *Manager {
	ctx, cancel := context.WithCancel(context.Background())
	m := &Manager{
		queue:      make(PriorityQueue, 0),
		processing: make(map[uuid.UUID]*Job),
		completed:  make(map[uuid.UUID]*Job),
		handlers:   make(map[string]JobHandler),
		workers:    workers,
		ctx:        ctx,
		cancel:     cancel,
	}
	m.cond = sync.NewCond(&m.mu)
	heap.Init(&m.queue)
	return m
}

// RegisterHandler registers a handler for a job type
func (m *Manager) RegisterHandler(jobType string, handler JobHandler) {
	m.mu.Lock()
	m.handlers[jobType] = handler
	m.mu.Unlock()
	log.Info().Str("type", jobType).Msg("Registered job handler")
}

// Enqueue adds a job to the queue
func (m *Manager) Enqueue(job *Job) {
	m.mu.Lock()
	job.CreatedAt = time.Now()
	job.Status = StatusPending
	if job.ID == uuid.Nil {
		job.ID = uuid.New()
	}
	if job.MaxRetries == 0 {
		job.MaxRetries = 3
	}
	heap.Push(&m.queue, job)
	m.mu.Unlock()
	m.cond.Signal()

	log.Info().
		Str("job_id", job.ID.String()).
		Str("type", job.Type).
		Str("priority", job.Priority.String()).
		Msg("Job enqueued")
}

// EnqueueWithData creates and enqueues a job
func (m *Manager) EnqueueWithData(jobType string, priority Priority, data interface{}) (uuid.UUID, error) {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return uuid.Nil, err
	}

	job := &Job{
		ID:       uuid.New(),
		Type:     jobType,
		Priority: priority,
		Data:     dataBytes,
	}

	m.Enqueue(job)
	return job.ID, nil
}

// Start starts the queue workers
func (m *Manager) Start() {
	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return
	}
	m.running = true
	m.mu.Unlock()

	for i := 0; i < m.workers; i++ {
		go m.worker(i)
	}

	log.Info().Int("workers", m.workers).Msg("Queue manager started")
}

// Stop stops the queue manager
func (m *Manager) Stop() {
	m.mu.Lock()
	m.running = false
	m.mu.Unlock()
	m.cancel()
	m.cond.Broadcast()
	log.Info().Msg("Queue manager stopped")
}

func (m *Manager) worker(id int) {
	log.Info().Int("worker_id", id).Msg("Worker started")

	for {
		m.mu.Lock()
		for m.running && m.queue.Len() == 0 {
			m.cond.Wait()
		}

		if !m.running {
			m.mu.Unlock()
			break
		}

		job := heap.Pop(&m.queue).(*Job)
		job.Status = StatusProcessing
		now := time.Now()
		job.StartedAt = &now
		m.processing[job.ID] = job
		m.mu.Unlock()

		m.processJob(id, job)
	}

	log.Info().Int("worker_id", id).Msg("Worker stopped")
}

func (m *Manager) processJob(workerID int, job *Job) {
	log.Info().
		Int("worker_id", workerID).
		Str("job_id", job.ID.String()).
		Str("type", job.Type).
		Msg("Processing job")

	m.mu.RLock()
	handler, ok := m.handlers[job.Type]
	m.mu.RUnlock()

	var result json.RawMessage
	var err error

	if !ok {
		err = &NoHandlerError{JobType: job.Type}
	} else {
		ctx, cancel := context.WithTimeout(m.ctx, 5*time.Minute)
		result, err = handler(ctx, job)
		cancel()
	}

	m.mu.Lock()
	now := time.Now()
	job.CompletedAt = &now
	delete(m.processing, job.ID)

	if err != nil {
		job.Retries++
		if job.Retries < job.MaxRetries {
			job.Status = StatusPending
			job.StartedAt = nil
			job.CompletedAt = nil
			heap.Push(&m.queue, job)
			m.mu.Unlock()
			m.cond.Signal()

			log.Warn().
				Str("job_id", job.ID.String()).
				Int("retry", job.Retries).
				Err(err).
				Msg("Job failed, retrying")
			return
		}

		job.Status = StatusFailed
		job.Error = err.Error()
		log.Error().
			Str("job_id", job.ID.String()).
			Err(err).
			Msg("Job failed permanently")
	} else {
		job.Status = StatusCompleted
		job.Result = result
		log.Info().
			Str("job_id", job.ID.String()).
			Msg("Job completed")
	}

	m.completed[job.ID] = job
	m.mu.Unlock()
}

// GetJob returns a job by ID
func (m *Manager) GetJob(id uuid.UUID) *Job {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if job, ok := m.processing[id]; ok {
		return job
	}
	if job, ok := m.completed[id]; ok {
		return job
	}

	for _, job := range m.queue {
		if job.ID == id {
			return job
		}
	}

	return nil
}

// GetStatus returns the status of a job
func (m *Manager) GetStatus(id uuid.UUID) (JobStatus, error) {
	job := m.GetJob(id)
	if job == nil {
		return "", &JobNotFoundError{ID: id}
	}
	return job.Status, nil
}

// CancelJob cancels a pending job
func (m *Manager) CancelJob(id uuid.UUID) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i, job := range m.queue {
		if job.ID == id {
			heap.Remove(&m.queue, i)
			job.Status = StatusCancelled
			now := time.Now()
			job.CompletedAt = &now
			m.completed[id] = job
			log.Info().Str("job_id", id.String()).Msg("Job cancelled")
			return nil
		}
	}

	return &JobNotFoundError{ID: id}
}

// GetStats returns queue statistics
func (m *Manager) GetStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pendingByPriority := make(map[string]int)
	for _, job := range m.queue {
		pendingByPriority[job.Priority.String()]++
	}

	completedCount := 0
	failedCount := 0
	for _, job := range m.completed {
		if job.Status == StatusCompleted {
			completedCount++
		} else if job.Status == StatusFailed {
			failedCount++
		}
	}

	return map[string]interface{}{
		"queued":     len(m.queue),
		"processing": len(m.processing),
		"completed":  completedCount,
		"failed":     failedCount,
		"by_priority": pendingByPriority,
		"workers":    m.workers,
		"running":    m.running,
	}
}

// CleanupCompleted removes old completed jobs
func (m *Manager) CleanupCompleted(olderThan time.Duration) int {
	m.mu.Lock()
	defer m.mu.Unlock()

	threshold := time.Now().Add(-olderThan)
	count := 0

	for id, job := range m.completed {
		if job.CompletedAt != nil && job.CompletedAt.Before(threshold) {
			delete(m.completed, id)
			count++
		}
	}

	if count > 0 {
		log.Info().Int("count", count).Msg("Cleaned up completed jobs")
	}

	return count
}

// NoHandlerError is returned when no handler is registered for a job type
type NoHandlerError struct {
	JobType string
}

func (e *NoHandlerError) Error() string {
	return "no handler registered for job type: " + e.JobType
}

// JobNotFoundError is returned when a job is not found
type JobNotFoundError struct {
	ID uuid.UUID
}

func (e *JobNotFoundError) Error() string {
	return "job not found: " + e.ID.String()
}




