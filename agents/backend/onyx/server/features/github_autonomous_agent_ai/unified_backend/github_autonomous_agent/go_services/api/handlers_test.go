package api

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/cache"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/search"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
)

func setupTestHandlers(t *testing.T) *Handlers {
	logger := zerolog.Nop()
	
	cacheService, err := cache.NewMultiTierCache(cache.Config{
		MemorySize:    1000,
		MemoryTTL:     5 * time.Minute,
		BadgerPath:    "/tmp/test_cache",
		EnableBadger:  false,
		EnableRedis:   false,
	})
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}

	searchIndex, err := search.NewIndex("/tmp/test_search")
	if err != nil {
		t.Fatalf("Failed to create search index: %v", err)
	}

	return NewHandlers(cacheService, searchIndex, logger)
}

func TestHealthHandler(t *testing.T) {
	handlers := setupTestHandlers(t)
	
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	
	handlers.HealthHandler(w, req)
	
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "healthy")
}

func TestCacheGetHandler(t *testing.T) {
	handlers := setupTestHandlers(t)
	ctx := context.Background()
	
	// Set a value first
	err := handlers.cacheService.Set(ctx, "test_key", "test_value", 5*time.Minute)
	assert.NoError(t, err)
	
	req := httptest.NewRequest(http.MethodGet, "/api/v1/cache?key=test_key", nil)
	w := httptest.NewRecorder()
	
	handlers.CacheGetHandler(w, req)
	
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "test_value")
}

func TestCacheSetHandler(t *testing.T) {
	handlers := setupTestHandlers(t)
	
	body := bytes.NewBufferString(`{"key":"test_key","value":"test_value"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/cache", body)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	handlers.CacheSetHandler(w, req)
	
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestSearchHandler(t *testing.T) {
	handlers := setupTestHandlers(t)
	
	req := httptest.NewRequest(http.MethodGet, "/api/v1/search?q=test", nil)
	w := httptest.NewRecorder()
	
	handlers.SearchHandler(w, req)
	
	assert.Equal(t, http.StatusOK, w.Code)
}












