// Package eventbus provides a high-performance pub/sub event system using channels
package eventbus

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
)

// Event represents an event in the system
type Event struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"`
	Data      json.RawMessage `json:"data"`
	Timestamp time.Time       `json:"timestamp"`
	Source    string          `json:"source,omitempty"`
}

// Handler is a function that handles events
type Handler func(event *Event) error

// AsyncHandler is an async handler that handles events
type AsyncHandler func(ctx context.Context, event *Event) error

// Subscription represents a subscription to an event type
type Subscription struct {
	ID        string
	EventType string
	Handler   Handler
	async     bool
}

// Bus is a high-performance event bus using Go channels
type Bus struct {
	handlers     map[string][]Subscription
	wildcardSubs []Subscription
	eventChan    chan *Event
	history      []*Event
	maxHistory   int
	mu           sync.RWMutex
	wg           sync.WaitGroup
	ctx          context.Context
	cancel       context.CancelFunc
}

// NewBus creates a new event bus
func NewBus(bufferSize int, maxHistory int) *Bus {
	ctx, cancel := context.WithCancel(context.Background())
	bus := &Bus{
		handlers:   make(map[string][]Subscription),
		eventChan:  make(chan *Event, bufferSize),
		history:    make([]*Event, 0),
		maxHistory: maxHistory,
		ctx:        ctx,
		cancel:     cancel,
	}

	go bus.processEvents()
	return bus
}

// Subscribe subscribes to an event type
func (b *Bus) Subscribe(eventType string, handler Handler) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	sub := Subscription{
		ID:        uuid.New().String(),
		EventType: eventType,
		Handler:   handler,
	}

	if eventType == "*" {
		b.wildcardSubs = append(b.wildcardSubs, sub)
	} else {
		b.handlers[eventType] = append(b.handlers[eventType], sub)
	}

	log.Debug().
		Str("subscription_id", sub.ID).
		Str("event_type", eventType).
		Msg("Event subscription added")

	return sub.ID
}

// SubscribeAsync subscribes with an async handler
func (b *Bus) SubscribeAsync(eventType string, handler AsyncHandler) string {
	wrappedHandler := func(event *Event) error {
		go func() {
			if err := handler(b.ctx, event); err != nil {
				log.Error().Err(err).Str("event_type", event.Type).Msg("Async handler error")
			}
		}()
		return nil
	}

	return b.Subscribe(eventType, wrappedHandler)
}

// Unsubscribe removes a subscription
func (b *Bus) Unsubscribe(subscriptionID string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	for eventType, subs := range b.handlers {
		for i, sub := range subs {
			if sub.ID == subscriptionID {
				b.handlers[eventType] = append(subs[:i], subs[i+1:]...)
				log.Debug().
					Str("subscription_id", subscriptionID).
					Msg("Event subscription removed")
				return true
			}
		}
	}

	for i, sub := range b.wildcardSubs {
		if sub.ID == subscriptionID {
			b.wildcardSubs = append(b.wildcardSubs[:i], b.wildcardSubs[i+1:]...)
			return true
		}
	}

	return false
}

// Publish publishes an event
func (b *Bus) Publish(eventType string, data interface{}) error {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}

	event := &Event{
		ID:        uuid.New().String(),
		Type:      eventType,
		Data:      dataBytes,
		Timestamp: time.Now(),
	}

	select {
	case b.eventChan <- event:
		return nil
	default:
		log.Warn().Str("event_type", eventType).Msg("Event bus buffer full, dropping event")
		return ErrBufferFull
	}
}

// PublishSync publishes an event and waits for handlers to complete
func (b *Bus) PublishSync(eventType string, data interface{}) error {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}

	event := &Event{
		ID:        uuid.New().String(),
		Type:      eventType,
		Data:      dataBytes,
		Timestamp: time.Now(),
	}

	b.addToHistory(event)
	return b.dispatchEvent(event)
}

func (b *Bus) processEvents() {
	for {
		select {
		case event := <-b.eventChan:
			b.addToHistory(event)
			if err := b.dispatchEvent(event); err != nil {
				log.Error().Err(err).Str("event_type", event.Type).Msg("Event dispatch error")
			}
		case <-b.ctx.Done():
			return
		}
	}
}

func (b *Bus) dispatchEvent(event *Event) error {
	b.mu.RLock()
	handlers := make([]Subscription, 0)
	handlers = append(handlers, b.handlers[event.Type]...)
	handlers = append(handlers, b.wildcardSubs...)
	b.mu.RUnlock()

	var lastErr error
	for _, sub := range handlers {
		if err := sub.Handler(event); err != nil {
			log.Error().
				Err(err).
				Str("subscription_id", sub.ID).
				Str("event_type", event.Type).
				Msg("Event handler error")
			lastErr = err
		}
	}

	log.Debug().
		Str("event_id", event.ID).
		Str("event_type", event.Type).
		Int("handlers", len(handlers)).
		Msg("Event dispatched")

	return lastErr
}

func (b *Bus) addToHistory(event *Event) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.history = append(b.history, event)
	if len(b.history) > b.maxHistory {
		b.history = b.history[1:]
	}
}

// GetHistory returns event history
func (b *Bus) GetHistory(eventType string, limit int) []*Event {
	b.mu.RLock()
	defer b.mu.RUnlock()

	var events []*Event
	for _, event := range b.history {
		if eventType == "" || event.Type == eventType {
			events = append(events, event)
		}
	}

	if limit > 0 && len(events) > limit {
		events = events[len(events)-limit:]
	}

	return events
}

// ClearHistory clears event history
func (b *Bus) ClearHistory() {
	b.mu.Lock()
	b.history = make([]*Event, 0)
	b.mu.Unlock()
}

// GetStats returns event bus statistics
func (b *Bus) GetStats() map[string]interface{} {
	b.mu.RLock()
	defer b.mu.RUnlock()

	handlerCount := 0
	for _, subs := range b.handlers {
		handlerCount += len(subs)
	}
	handlerCount += len(b.wildcardSubs)

	return map[string]interface{}{
		"subscriptions":      handlerCount,
		"event_types":        len(b.handlers),
		"history_size":       len(b.history),
		"buffer_size":        cap(b.eventChan),
		"pending_events":     len(b.eventChan),
	}
}

// Close shuts down the event bus
func (b *Bus) Close() {
	b.cancel()
	close(b.eventChan)
}

// Wait waits for all events to be processed
func (b *Bus) Wait() {
	b.wg.Wait()
}

// ErrBufferFull is returned when the event buffer is full
var ErrBufferFull = &BufferFullError{}

type BufferFullError struct{}

func (e *BufferFullError) Error() string {
	return "event bus buffer is full"
}

// Common event types
const (
	EventVideoCreated     = "video.created"
	EventVideoStarted     = "video.started"
	EventVideoProgress    = "video.progress"
	EventVideoCompleted   = "video.completed"
	EventVideoFailed      = "video.failed"
	EventScriptProcessed  = "script.processed"
	EventImagesGenerated  = "images.generated"
	EventAudioGenerated   = "audio.generated"
	EventSubtitlesCreated = "subtitles.created"
	EventJobEnqueued      = "job.enqueued"
	EventJobStarted       = "job.started"
	EventJobCompleted     = "job.completed"
	EventJobFailed        = "job.failed"
)

// VideoEvent represents a video-related event
type VideoEvent struct {
	VideoID     string  `json:"video_id"`
	Status      string  `json:"status"`
	Progress    float64 `json:"progress,omitempty"`
	CurrentStep string  `json:"current_step,omitempty"`
	Error       string  `json:"error,omitempty"`
	VideoURL    string  `json:"video_url,omitempty"`
}

// JobEvent represents a job-related event
type JobEvent struct {
	JobID   string `json:"job_id"`
	JobType string `json:"job_type"`
	Status  string `json:"status"`
	Error   string `json:"error,omitempty"`
}




