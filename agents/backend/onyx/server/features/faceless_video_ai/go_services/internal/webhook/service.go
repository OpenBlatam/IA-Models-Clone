// Package webhook provides a high-performance webhook notification service
package webhook

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
)

// Config holds webhook service configuration
type Config struct {
	Timeout       time.Duration
	MaxRetries    int
	RetryDelay    time.Duration
	WorkerCount   int
	QueueSize     int
}

// DefaultConfig returns default webhook configuration
func DefaultConfig() Config {
	return Config{
		Timeout:     10 * time.Second,
		MaxRetries:  3,
		RetryDelay:  2 * time.Second,
		WorkerCount: 5,
		QueueSize:   1000,
	}
}

// Payload represents a webhook payload
type Payload struct {
	VideoID   string                 `json:"video_id"`
	Status    string                 `json:"status"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// CompletionPayload represents a completion notification payload
type CompletionPayload struct {
	VideoID   string    `json:"video_id"`
	Status    string    `json:"status"`
	VideoURL  string    `json:"video_url"`
	Duration  float64   `json:"duration"`
	FileSize  int64     `json:"file_size"`
	Timestamp time.Time `json:"timestamp"`
}

// FailurePayload represents a failure notification payload
type FailurePayload struct {
	VideoID   string    `json:"video_id"`
	Status    string    `json:"status"`
	Error     string    `json:"error"`
	Timestamp time.Time `json:"timestamp"`
}

// webhookJob represents a webhook delivery job
type webhookJob struct {
	URL       string
	Payload   interface{}
	Retries   int
	VideoID   uuid.UUID
}

// Service manages webhook notifications
type Service struct {
	config     Config
	client     *http.Client
	urls       map[uuid.UUID][]string
	jobQueue   chan *webhookJob
	stats      *Stats
	mu         sync.RWMutex
	ctx        context.Context
	cancel     context.CancelFunc
}

// Stats holds webhook service statistics
type Stats struct {
	TotalSent     int64
	TotalFailed   int64
	TotalRetried  int64
	AverageLatency time.Duration
	mu            sync.RWMutex
}

// NewService creates a new webhook service
func NewService(config Config) *Service {
	ctx, cancel := context.WithCancel(context.Background())

	s := &Service{
		config: config,
		client: &http.Client{
			Timeout: config.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		urls:     make(map[uuid.UUID][]string),
		jobQueue: make(chan *webhookJob, config.QueueSize),
		stats:    &Stats{},
		ctx:      ctx,
		cancel:   cancel,
	}

	for i := 0; i < config.WorkerCount; i++ {
		go s.worker(i)
	}

	return s
}

// Register registers a webhook URL for a video
func (s *Service) Register(videoID uuid.UUID, url string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.urls[videoID]; !ok {
		s.urls[videoID] = make([]string, 0)
	}

	for _, existingURL := range s.urls[videoID] {
		if existingURL == url {
			return
		}
	}

	s.urls[videoID] = append(s.urls[videoID], url)
	log.Info().
		Str("video_id", videoID.String()).
		Str("url", url).
		Msg("Webhook registered")
}

// Unregister removes a webhook URL
func (s *Service) Unregister(videoID uuid.UUID, url string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if urls, ok := s.urls[videoID]; ok {
		for i, existingURL := range urls {
			if existingURL == url {
				s.urls[videoID] = append(urls[:i], urls[i+1:]...)
				break
			}
		}
		if len(s.urls[videoID]) == 0 {
			delete(s.urls, videoID)
		}
	}
}

// UnregisterAll removes all webhook URLs for a video
func (s *Service) UnregisterAll(videoID uuid.UUID) {
	s.mu.Lock()
	delete(s.urls, videoID)
	s.mu.Unlock()
}

// Send sends a webhook notification
func (s *Service) Send(videoID uuid.UUID, status string, data map[string]interface{}) error {
	payload := &Payload{
		VideoID:   videoID.String(),
		Status:    status,
		Timestamp: time.Now(),
		Data:      data,
	}

	return s.sendToVideo(videoID, payload)
}

// NotifyCompletion sends a completion notification
func (s *Service) NotifyCompletion(videoID uuid.UUID, videoURL string, duration float64, fileSize int64) error {
	payload := &CompletionPayload{
		VideoID:   videoID.String(),
		Status:    "completed",
		VideoURL:  videoURL,
		Duration:  duration,
		FileSize:  fileSize,
		Timestamp: time.Now(),
	}

	return s.sendToVideo(videoID, payload)
}

// NotifyFailure sends a failure notification
func (s *Service) NotifyFailure(videoID uuid.UUID, err string) error {
	payload := &FailurePayload{
		VideoID:   videoID.String(),
		Status:    "failed",
		Error:     err,
		Timestamp: time.Now(),
	}

	return s.sendToVideo(videoID, payload)
}

func (s *Service) sendToVideo(videoID uuid.UUID, payload interface{}) error {
	s.mu.RLock()
	urls := s.urls[videoID]
	s.mu.RUnlock()

	if len(urls) == 0 {
		return nil
	}

	for _, url := range urls {
		job := &webhookJob{
			URL:     url,
			Payload: payload,
			VideoID: videoID,
		}

		select {
		case s.jobQueue <- job:
		default:
			log.Warn().
				Str("video_id", videoID.String()).
				Str("url", url).
				Msg("Webhook queue full, dropping notification")
		}
	}

	return nil
}

func (s *Service) worker(id int) {
	log.Debug().Int("worker_id", id).Msg("Webhook worker started")

	for {
		select {
		case job := <-s.jobQueue:
			s.processJob(job)
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *Service) processJob(job *webhookJob) {
	start := time.Now()

	err := s.sendWebhook(job.URL, job.Payload)
	if err != nil {
		job.Retries++
		if job.Retries <= s.config.MaxRetries {
			s.stats.mu.Lock()
			s.stats.TotalRetried++
			s.stats.mu.Unlock()

			log.Warn().
				Str("url", job.URL).
				Int("retry", job.Retries).
				Err(err).
				Msg("Webhook failed, retrying")

			time.AfterFunc(s.config.RetryDelay*time.Duration(job.Retries), func() {
				select {
				case s.jobQueue <- job:
				default:
					log.Error().Str("url", job.URL).Msg("Failed to retry webhook, queue full")
				}
			})
			return
		}

		s.stats.mu.Lock()
		s.stats.TotalFailed++
		s.stats.mu.Unlock()

		log.Error().
			Str("url", job.URL).
			Err(err).
			Msg("Webhook failed permanently")
		return
	}

	latency := time.Since(start)
	s.stats.mu.Lock()
	s.stats.TotalSent++
	s.stats.AverageLatency = (s.stats.AverageLatency*time.Duration(s.stats.TotalSent-1) + latency) / time.Duration(s.stats.TotalSent)
	s.stats.mu.Unlock()

	log.Debug().
		Str("url", job.URL).
		Dur("latency", latency).
		Msg("Webhook sent successfully")
}

func (s *Service) sendWebhook(url string, payload interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(s.ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "FacelessVideoAI-Webhook/1.0")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("webhook returned status %d", resp.StatusCode)
	}

	return nil
}

// GetStats returns webhook service statistics
func (s *Service) GetStats() map[string]interface{} {
	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()

	s.mu.RLock()
	registeredCount := 0
	for _, urls := range s.urls {
		registeredCount += len(urls)
	}
	s.mu.RUnlock()

	return map[string]interface{}{
		"total_sent":      s.stats.TotalSent,
		"total_failed":    s.stats.TotalFailed,
		"total_retried":   s.stats.TotalRetried,
		"average_latency": s.stats.AverageLatency.String(),
		"registered_urls": registeredCount,
		"queue_size":      len(s.jobQueue),
	}
}

// Close shuts down the webhook service
func (s *Service) Close() {
	s.cancel()
	close(s.jobQueue)
}




