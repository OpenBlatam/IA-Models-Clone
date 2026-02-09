package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/semaphore"
	"golang.org/x/time/rate"
)

const (
	BaseURL           = "https://openrouter.ai/api/v1"
	DefaultModel      = "anthropic/claude-3.5-sonnet"
	DefaultMaxTokens  = 4096
	DefaultTimeout    = 60 * time.Second
	MaxRetries        = 3
	RetryDelay        = 1 * time.Second
)

type Client struct {
	apiKey      string
	httpClient  *http.Client
	limiter     *rate.Limiter
	semaphore   *semaphore.Weighted
	mu          sync.RWMutex
	stats       *Stats
}

type Stats struct {
	TotalRequests     int64
	SuccessfulRequests int64
	FailedRequests    int64
	TotalTokensUsed   int64
	TotalLatencyMs    int64
	mu                sync.Mutex
}

type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []Message     `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type AnalysisResult struct {
	Framework      string            `json:"framework"`
	Structure      []string          `json:"structure"`
	Keywords       []string          `json:"keywords"`
	Summary        string            `json:"summary"`
	Tone           string            `json:"tone"`
	Audience       string            `json:"audience"`
	Hashtags       []string          `json:"hashtags"`
	ContentLength  string            `json:"content_length"`
	Metadata       map[string]interface{} `json:"metadata"`
}

type VariantResult struct {
	Original   string   `json:"original"`
	Variants   []string `json:"variants"`
	Framework  string   `json:"framework"`
	Tone       string   `json:"tone"`
}

func NewClient(apiKey string) *Client {
	return &Client{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: DefaultTimeout,
		},
		limiter:   rate.NewLimiter(rate.Every(time.Second), 10),
		semaphore: semaphore.NewWeighted(10),
		stats:     &Stats{},
	}
}

func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	if req.Model == "" {
		req.Model = DefaultModel
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = DefaultMaxTokens
	}

	if err := c.limiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit wait: %w", err)
	}

	if err := c.semaphore.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("semaphore acquire: %w", err)
	}
	defer c.semaphore.Release(1)

	start := time.Now()
	c.stats.mu.Lock()
	c.stats.TotalRequests++
	c.stats.mu.Unlock()

	var lastErr error
	for attempt := 0; attempt < MaxRetries; attempt++ {
		if attempt > 0 {
			delay := RetryDelay * time.Duration(1<<uint(attempt-1))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		resp, err := c.doRequest(ctx, req)
		if err == nil {
			latency := time.Since(start).Milliseconds()
			c.stats.mu.Lock()
			c.stats.SuccessfulRequests++
			c.stats.TotalLatencyMs += latency
			if resp.Usage.TotalTokens > 0 {
				c.stats.TotalTokensUsed += int64(resp.Usage.TotalTokens)
			}
			c.stats.mu.Unlock()
			return resp, nil
		}

		lastErr = err
		log.Warn().Err(err).Int("attempt", attempt+1).Msg("Request failed, retrying")
	}

	c.stats.mu.Lock()
	c.stats.FailedRequests++
	c.stats.mu.Unlock()

	return nil, fmt.Errorf("all retries failed: %w", lastErr)
}

func (c *Client) doRequest(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("HTTP-Referer", "https://blatam-academy.com")
	httpReq.Header.Set("X-Title", "Social Video Transcriber AI")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func (c *Client) AnalyzeContent(ctx context.Context, text string) (*AnalysisResult, error) {
	prompt := fmt.Sprintf(`Analyze the following content and return a JSON object with:
- framework: The content framework used (Hook-Story-Offer, AIDA, PAS, STAR, BAB, etc.)
- structure: Array of structural elements
- keywords: Array of key terms
- summary: Brief summary
- tone: Detected tone
- audience: Target audience
- hashtags: Suggested hashtags
- content_length: short/medium/long

Content:
%s

Return ONLY valid JSON, no markdown.`, text)

	req := ChatRequest{
		Model:       DefaultModel,
		Temperature: 0.3,
		Messages: []Message{
			{Role: "system", Content: "You are a content analysis expert. Always respond with valid JSON only."},
			{Role: "user", Content: prompt},
		},
	}

	resp, err := c.Chat(ctx, req)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices")
	}

	var result AnalysisResult
	content := resp.Choices[0].Message.Content
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return nil, fmt.Errorf("parse analysis result: %w", err)
	}

	return &result, nil
}

func (c *Client) GenerateVariants(ctx context.Context, text string, count int, preserveFramework bool) (*VariantResult, error) {
	frameworkNote := ""
	if preserveFramework {
		frameworkNote = "Preserve the same content framework structure."
	}

	prompt := fmt.Sprintf(`Generate %d unique variants of the following content.
%s
Keep the same meaning and approximate length.

Original content:
%s

Return a JSON object with:
- original: The original text
- variants: Array of %d variant texts
- framework: The detected framework
- tone: The detected tone

Return ONLY valid JSON, no markdown.`, count, frameworkNote, text, count)

	req := ChatRequest{
		Model:       DefaultModel,
		Temperature: 0.7,
		Messages: []Message{
			{Role: "system", Content: "You are a content creation expert. Always respond with valid JSON only."},
			{Role: "user", Content: prompt},
		},
	}

	resp, err := c.Chat(ctx, req)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices")
	}

	var result VariantResult
	content := resp.Choices[0].Message.Content
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return nil, fmt.Errorf("parse variants result: %w", err)
	}

	return &result, nil
}

func (c *Client) Summarize(ctx context.Context, text string, style string) (string, error) {
	var stylePrompt string
	switch style {
	case "brief":
		stylePrompt = "Create a very brief 1-2 sentence summary."
	case "detailed":
		stylePrompt = "Create a detailed paragraph summary."
	case "bullets":
		stylePrompt = "Create a bullet-point summary with key points."
	default:
		stylePrompt = "Create a concise summary."
	}

	prompt := fmt.Sprintf(`%s

Content:
%s

Return only the summary text, no JSON or markdown.`, stylePrompt, text)

	req := ChatRequest{
		Model:       DefaultModel,
		Temperature: 0.3,
		Messages: []Message{
			{Role: "system", Content: "You are a summarization expert."},
			{Role: "user", Content: prompt},
		},
	}

	resp, err := c.Chat(ctx, req)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices")
	}

	return resp.Choices[0].Message.Content, nil
}

func (c *Client) ExtractKeywords(ctx context.Context, text string, maxKeywords int) ([]string, error) {
	prompt := fmt.Sprintf(`Extract the %d most important keywords from this content.
Return a JSON array of strings only.

Content:
%s`, maxKeywords, text)

	req := ChatRequest{
		Model:       DefaultModel,
		Temperature: 0.2,
		Messages: []Message{
			{Role: "system", Content: "You extract keywords. Respond with JSON array only."},
			{Role: "user", Content: prompt},
		},
	}

	resp, err := c.Chat(ctx, req)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices")
	}

	var keywords []string
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &keywords); err != nil {
		return nil, fmt.Errorf("parse keywords: %w", err)
	}

	return keywords, nil
}

func (c *Client) Translate(ctx context.Context, text, targetLang string) (string, error) {
	prompt := fmt.Sprintf(`Translate the following text to %s.
Keep the same tone and style.
Return only the translated text.

Text:
%s`, targetLang, text)

	req := ChatRequest{
		Model:       DefaultModel,
		Temperature: 0.2,
		Messages: []Message{
			{Role: "system", Content: "You are a professional translator."},
			{Role: "user", Content: prompt},
		},
	}

	resp, err := c.Chat(ctx, req)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices")
	}

	return resp.Choices[0].Message.Content, nil
}

func (c *Client) GetStats() map[string]interface{} {
	c.stats.mu.Lock()
	defer c.stats.mu.Unlock()

	avgLatency := float64(0)
	if c.stats.SuccessfulRequests > 0 {
		avgLatency = float64(c.stats.TotalLatencyMs) / float64(c.stats.SuccessfulRequests)
	}

	return map[string]interface{}{
		"total_requests":      c.stats.TotalRequests,
		"successful_requests": c.stats.SuccessfulRequests,
		"failed_requests":     c.stats.FailedRequests,
		"total_tokens_used":   c.stats.TotalTokensUsed,
		"avg_latency_ms":      avgLatency,
		"success_rate":        float64(c.stats.SuccessfulRequests) / float64(max(c.stats.TotalRequests, 1)) * 100,
	}
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}












