// Package openrouter provides a high-performance client for OpenRouter API
// OpenRouter unifies access to multiple LLM providers (OpenAI, Claude, Gemini, etc.)
package openrouter

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	DefaultBaseURL     = "https://openrouter.ai/api/v1"
	DefaultTimeout     = 60 * time.Second
	DefaultMaxRetries  = 3
	DefaultRetryDelay  = 1 * time.Second
)

// Model represents available models through OpenRouter
type Model string

const (
	ModelGPT4              Model = "openai/gpt-4"
	ModelGPT4Turbo         Model = "openai/gpt-4-turbo"
	ModelGPT35Turbo        Model = "openai/gpt-3.5-turbo"
	ModelClaude3Opus       Model = "anthropic/claude-3-opus"
	ModelClaude3Sonnet     Model = "anthropic/claude-3-sonnet"
	ModelClaude3Haiku      Model = "anthropic/claude-3-haiku"
	ModelGeminiPro         Model = "google/gemini-pro"
	ModelGemini15Pro       Model = "google/gemini-1.5-pro"
	ModelMistralLarge      Model = "mistralai/mistral-large"
	ModelMistralMedium     Model = "mistralai/mistral-medium"
	ModelLlama370B         Model = "meta-llama/llama-3-70b-instruct"
	ModelCommandRPlus      Model = "cohere/command-r-plus"
)

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents a chat completion request
type ChatRequest struct {
	Model       Model     `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

// ChatResponse represents a chat completion response
type ChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// StreamChunk represents a streaming response chunk
type StreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// ClientConfig holds configuration for the OpenRouter client
type ClientConfig struct {
	APIKey      string
	BaseURL     string
	Timeout     time.Duration
	MaxRetries  int
	RetryDelay  time.Duration
	HTTPClient  *http.Client
	AppName     string
	AppURL      string
}

// Client is a high-performance OpenRouter API client
type Client struct {
	config     ClientConfig
	httpClient *http.Client
	mu         sync.RWMutex
	stats      *Stats
}

// Stats tracks client statistics
type Stats struct {
	TotalRequests     int64
	SuccessfulRequests int64
	FailedRequests     int64
	TotalTokensUsed   int64
	AverageLatency    time.Duration
	mu                sync.RWMutex
}

// NewClient creates a new OpenRouter client
func NewClient(config ClientConfig) *Client {
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	if config.Timeout == 0 {
		config.Timeout = DefaultTimeout
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = DefaultMaxRetries
	}
	if config.RetryDelay == 0 {
		config.RetryDelay = DefaultRetryDelay
	}

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: config.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		}
	}

	return &Client{
		config:     config,
		httpClient: httpClient,
		stats:      &Stats{},
	}
}

// Chat sends a chat completion request
func (c *Client) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	start := time.Now()
	c.stats.incrementRequests()

	var resp *ChatResponse
	var err error

	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		resp, err = c.doChat(ctx, req)
		if err == nil {
			c.stats.recordSuccess(time.Since(start))
			if resp.Usage.TotalTokens > 0 {
				c.stats.addTokens(int64(resp.Usage.TotalTokens))
			}
			return resp, nil
		}

		if !isRetryable(err) {
			break
		}

		if attempt < c.config.MaxRetries {
			time.Sleep(c.config.RetryDelay * time.Duration(attempt+1))
		}
	}

	c.stats.recordFailure()
	return nil, err
}

func (c *Client) doChat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.config.BaseURL+"/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	c.setHeaders(httpReq)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(httpResp.Body)
		return nil, &APIError{
			StatusCode: httpResp.StatusCode,
			Message:    string(bodyBytes),
		}
	}

	var resp ChatResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// ChatStream sends a streaming chat completion request
func (c *Client) ChatStream(ctx context.Context, req *ChatRequest) (<-chan StreamChunk, <-chan error) {
	chunkChan := make(chan StreamChunk, 100)
	errChan := make(chan error, 1)

	req.Stream = true

	go func() {
		defer close(chunkChan)
		defer close(errChan)

		body, err := json.Marshal(req)
		if err != nil {
			errChan <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		httpReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			c.config.BaseURL+"/chat/completions",
			bytes.NewReader(body),
		)
		if err != nil {
			errChan <- fmt.Errorf("failed to create request: %w", err)
			return
		}

		c.setHeaders(httpReq)

		httpResp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errChan <- fmt.Errorf("request failed: %w", err)
			return
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(httpResp.Body)
			errChan <- &APIError{
				StatusCode: httpResp.StatusCode,
				Message:    string(bodyBytes),
			}
			return
		}

		reader := bufio.NewReader(httpResp.Body)
		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err != io.EOF {
					errChan <- err
				}
				return
			}

			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}

			data := bytes.TrimPrefix(line, []byte("data: "))
			if string(data) == "[DONE]" {
				return
			}

			var chunk StreamChunk
			if err := json.Unmarshal(data, &chunk); err != nil {
				continue
			}

			select {
			case chunkChan <- chunk:
			case <-ctx.Done():
				return
			}
		}
	}()

	return chunkChan, errChan
}

// EnhanceScript enhances a script using the specified model
func (c *Client) EnhanceScript(ctx context.Context, script, language string, model Model) (string, error) {
	if model == "" {
		model = ModelClaude3Sonnet
	}

	prompt := fmt.Sprintf(`You are a professional video script enhancer.
Enhance the following script to make it more engaging, clear, and suitable for video narration.
Keep the original meaning and style.
Language: %s

Script:
%s

Enhanced script:`, language, script)

	resp, err := c.Chat(ctx, &ChatRequest{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
		MaxTokens:   2000,
		Temperature: 0.7,
	})
	if err != nil {
		return script, err
	}

	if len(resp.Choices) > 0 {
		return resp.Choices[0].Message.Content, nil
	}

	return script, nil
}

// GenerateImagePrompt generates an image prompt from text
func (c *Client) GenerateImagePrompt(ctx context.Context, text, style string, model Model) (string, error) {
	if model == "" {
		model = ModelGPT4Turbo
	}

	prompt := fmt.Sprintf(`Generate a detailed image generation prompt for the following text.
The image should visualize the concept in a %s style.
Only output the image prompt, nothing else.

Text: %s

Image prompt:`, style, text)

	resp, err := c.Chat(ctx, &ChatRequest{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
		MaxTokens:   500,
		Temperature: 0.8,
	})
	if err != nil {
		return "", err
	}

	if len(resp.Choices) > 0 {
		return resp.Choices[0].Message.Content, nil
	}

	return "", nil
}

// GetModels returns available models
func (c *Client) GetModels(ctx context.Context) ([]ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		c.config.BaseURL+"/models",
		nil,
	)
	if err != nil {
		return nil, err
	}

	c.setHeaders(httpReq)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	var response struct {
		Data []ModelInfo `json:"data"`
	}
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, err
	}

	return response.Data, nil
}

// ModelInfo contains information about a model
type ModelInfo struct {
	ID            string  `json:"id"`
	Name          string  `json:"name"`
	Description   string  `json:"description"`
	Pricing       Pricing `json:"pricing"`
	ContextLength int     `json:"context_length"`
}

// Pricing contains model pricing information
type Pricing struct {
	Prompt     float64 `json:"prompt"`
	Completion float64 `json:"completion"`
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	if c.config.AppName != "" {
		req.Header.Set("HTTP-Referer", c.config.AppURL)
		req.Header.Set("X-Title", c.config.AppName)
	}
}

// GetStats returns client statistics
func (c *Client) GetStats() Stats {
	c.stats.mu.RLock()
	defer c.stats.mu.RUnlock()
	return *c.stats
}

func (s *Stats) incrementRequests() {
	s.mu.Lock()
	s.TotalRequests++
	s.mu.Unlock()
}

func (s *Stats) recordSuccess(latency time.Duration) {
	s.mu.Lock()
	s.SuccessfulRequests++
	s.AverageLatency = (s.AverageLatency*time.Duration(s.SuccessfulRequests-1) + latency) / time.Duration(s.SuccessfulRequests)
	s.mu.Unlock()
}

func (s *Stats) recordFailure() {
	s.mu.Lock()
	s.FailedRequests++
	s.mu.Unlock()
}

func (s *Stats) addTokens(tokens int64) {
	s.mu.Lock()
	s.TotalTokensUsed += tokens
	s.mu.Unlock()
}

// APIError represents an API error
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API error (status %d): %s", e.StatusCode, e.Message)
}

func isRetryable(err error) bool {
	if apiErr, ok := err.(*APIError); ok {
		return apiErr.StatusCode >= 500 || apiErr.StatusCode == 429
	}
	return true
}




