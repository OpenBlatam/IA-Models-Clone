// Package websocket provides a high-performance WebSocket manager
// using goroutines for concurrent connection handling
package websocket

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog/log"
)

const (
	writeWait      = 10 * time.Second
	pongWait       = 60 * time.Second
	pingPeriod     = (pongWait * 9) / 10
	maxMessageSize = 512 * 1024
)

// Message represents a WebSocket message
type Message struct {
	Type      string          `json:"type"`
	VideoID   string          `json:"video_id,omitempty"`
	Data      json.RawMessage `json:"data,omitempty"`
	Timestamp time.Time       `json:"timestamp"`
}

// ProgressUpdate represents a video generation progress update
type ProgressUpdate struct {
	VideoID     string  `json:"video_id"`
	Status      string  `json:"status"`
	Progress    float64 `json:"progress"`
	CurrentStep string  `json:"current_step"`
	Message     string  `json:"message,omitempty"`
}

// Client represents a WebSocket client connection
type Client struct {
	ID       string
	VideoID  uuid.UUID
	conn     *websocket.Conn
	send     chan []byte
	manager  *Manager
	mu       sync.Mutex
}

// Manager manages WebSocket connections
type Manager struct {
	clients    map[*Client]bool
	videoSubs  map[uuid.UUID]map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

// NewManager creates a new WebSocket manager
func NewManager() *Manager {
	return &Manager{
		clients:    make(map[*Client]bool),
		videoSubs:  make(map[uuid.UUID]map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

// Run starts the WebSocket manager event loop
func (m *Manager) Run() {
	for {
		select {
		case client := <-m.register:
			m.mu.Lock()
			m.clients[client] = true
			if _, ok := m.videoSubs[client.VideoID]; !ok {
				m.videoSubs[client.VideoID] = make(map[*Client]bool)
			}
			m.videoSubs[client.VideoID][client] = true
			m.mu.Unlock()
			log.Info().
				Str("client_id", client.ID).
				Str("video_id", client.VideoID.String()).
				Msg("Client connected")

		case client := <-m.unregister:
			m.mu.Lock()
			if _, ok := m.clients[client]; ok {
				delete(m.clients, client)
				if subs, ok := m.videoSubs[client.VideoID]; ok {
					delete(subs, client)
					if len(subs) == 0 {
						delete(m.videoSubs, client.VideoID)
					}
				}
				close(client.send)
			}
			m.mu.Unlock()
			log.Info().
				Str("client_id", client.ID).
				Msg("Client disconnected")

		case message := <-m.broadcast:
			m.mu.RLock()
			for client := range m.clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(m.clients, client)
				}
			}
			m.mu.RUnlock()
		}
	}
}

// RegisterClient registers a new client
func (m *Manager) RegisterClient(conn *websocket.Conn, videoID uuid.UUID) *Client {
	client := &Client{
		ID:      uuid.New().String(),
		VideoID: videoID,
		conn:    conn,
		send:    make(chan []byte, 256),
		manager: m,
	}

	m.register <- client
	return client
}

// UnregisterClient unregisters a client
func (m *Manager) UnregisterClient(client *Client) {
	m.unregister <- client
}

// SendToVideo sends a message to all clients subscribed to a video
func (m *Manager) SendToVideo(videoID uuid.UUID, msg *Message) error {
	msg.Timestamp = time.Now()
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	m.mu.RLock()
	clients, ok := m.videoSubs[videoID]
	m.mu.RUnlock()

	if !ok {
		return nil
	}

	for client := range clients {
		select {
		case client.send <- data:
		default:
			m.unregister <- client
		}
	}

	return nil
}

// SendProgressUpdate sends a progress update for a video
func (m *Manager) SendProgressUpdate(update *ProgressUpdate) error {
	videoID, err := uuid.Parse(update.VideoID)
	if err != nil {
		return err
	}

	data, err := json.Marshal(update)
	if err != nil {
		return err
	}

	msg := &Message{
		Type:    "progress",
		VideoID: update.VideoID,
		Data:    data,
	}

	return m.SendToVideo(videoID, msg)
}

// Broadcast sends a message to all connected clients
func (m *Manager) Broadcast(msg *Message) error {
	msg.Timestamp = time.Now()
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	m.broadcast <- data
	return nil
}

// GetStats returns manager statistics
func (m *Manager) GetStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"total_clients":   len(m.clients),
		"video_subscriptions": len(m.videoSubs),
	}
}

// ReadPump reads messages from the client
func (c *Client) ReadPump() {
	defer func() {
		c.manager.UnregisterClient(c)
		c.conn.Close()
	}()

	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Error().Err(err).Str("client_id", c.ID).Msg("WebSocket read error")
			}
			break
		}

		var msg Message
		if err := json.Unmarshal(message, &msg); err != nil {
			continue
		}

		c.handleMessage(&msg)
	}
}

// WritePump writes messages to the client
func (c *Client) WritePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (c *Client) handleMessage(msg *Message) {
	switch msg.Type {
	case "ping":
		response := &Message{
			Type:      "pong",
			Timestamp: time.Now(),
		}
		data, _ := json.Marshal(response)
		c.send <- data

	case "subscribe":
		log.Info().
			Str("client_id", c.ID).
			Str("type", msg.Type).
			Msg("Client subscription request")

	default:
		log.Debug().
			Str("client_id", c.ID).
			Str("type", msg.Type).
			Msg("Unknown message type")
	}
}

// SendMessage sends a message to the client
func (c *Client) SendMessage(msg *Message) error {
	msg.Timestamp = time.Now()
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	select {
	case c.send <- data:
		return nil
	default:
		return ErrSendBufferFull
	}
}

// ErrSendBufferFull is returned when the send buffer is full
var ErrSendBufferFull = &SendBufferFullError{}

type SendBufferFullError struct{}

func (e *SendBufferFullError) Error() string {
	return "send buffer is full"
}




