package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/cache"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/git"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/search"
	"github.com/rs/zerolog"
)

// Handlers contains all HTTP handlers
type Handlers struct {
	cacheService *cache.MultiTierCache
	searchIndex  *search.Index
	logger       zerolog.Logger
	validator    *Validator
}

// NewHandlers creates a new handlers instance
func NewHandlers(cacheService *cache.MultiTierCache, searchIndex *search.Index, logger zerolog.Logger) *Handlers {
	return &Handlers{
		cacheService: cacheService,
		searchIndex:  searchIndex,
		logger:       logger,
		validator:    NewValidator(),
	}
}

// HealthHandler handles health check requests
func (h *Handlers) HealthHandler(w http.ResponseWriter, r *http.Request) {
	healthData := map[string]interface{}{
		"service": "go-services",
		"status":  "healthy",
		"time":    time.Now().UTC().Format(time.RFC3339),
	}
	
	h.writeSuccess(w, http.StatusOK, healthData)
}

// GitCloneHandler handles Git clone requests
func (h *Handlers) GitCloneHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Validate request
	if err := h.validator.ValidateGitCloneRequest(r); err != nil {
		h.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	var req struct {
		URL  string `json:"url"`
		Path string `json:"path"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Fallback to query parameters
		req.URL = r.URL.Query().Get("url")
		req.Path = r.URL.Query().Get("path")
	}

	if err := git.Clone(req.URL, req.Path, nil); err != nil {
		h.logger.Error().Err(err).Str("url", req.URL).Msg("Clone failed")
		h.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

		h.writeSuccess(w, http.StatusOK, map[string]interface{}{
			"path": req.Path,
		})
}

// SearchHandler handles search requests
func (h *Handlers) SearchHandler(w http.ResponseWriter, r *http.Request) {
	// Validate request
	if err := h.validator.ValidateSearchRequest(r); err != nil {
		h.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	query := r.URL.Query().Get("q")

	limit := h.getIntQuery(r, "limit", 10)
	offset := h.getIntQuery(r, "offset", 0)

	results, err := h.searchIndex.Search(query, search.SearchOptions{
		Limit:  limit,
		Offset:  offset,
		Explain: false,
	})
	if err != nil {
		h.logger.Error().Err(err).Str("query", query).Msg("Search failed")
		h.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

		h.writeSuccess(w, http.StatusOK, map[string]interface{}{
			"total":    results.Total,
			"hits":     len(results.Hits),
			"duration": results.Duration.Milliseconds(),
			"results":  results.Hits,
		})
}

// CacheGetHandler handles cache get requests
func (h *Handlers) CacheGetHandler(w http.ResponseWriter, r *http.Request) {
	// Validate request
	if err := h.validator.ValidateCacheRequest(r); err != nil {
		h.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	key := r.URL.Query().Get("key")

	val, found := h.cacheService.Get(r.Context(), key)
	if !found {
		h.writeError(w, http.StatusNotFound, "Key not found")
		return
	}

		h.writeSuccess(w, http.StatusOK, map[string]interface{}{
			"key":   key,
			"value": val,
		})
}

// CacheSetHandler handles cache set requests
func (h *Handlers) CacheSetHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Key   string      `json:"key"`
		Value interface{} `json:"value"`
		TTL   int         `json:"ttl,omitempty"` // TTL in seconds
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Fallback to query parameters
		req.Key = r.URL.Query().Get("key")
		req.Value = r.URL.Query().Get("value")
	}

	if req.Key == "" || req.Value == nil {
		h.writeError(w, http.StatusBadRequest, "Missing key or value parameter")
		return
	}

	ttl := 5 * time.Minute
	if req.TTL > 0 {
		ttl = time.Duration(req.TTL) * time.Second
	}

	if err := h.cacheService.Set(r.Context(), req.Key, req.Value, ttl); err != nil {
		h.logger.Error().Err(err).Msg("Cache set failed")
		h.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

		h.writeSuccess(w, http.StatusOK, map[string]interface{}{
			"key": req.Key,
		})
}

// CacheDeleteHandler handles cache delete requests
func (h *Handlers) CacheDeleteHandler(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	if key == "" {
		h.writeError(w, http.StatusBadRequest, "Missing key parameter")
		return
	}

	if err := h.cacheService.Delete(r.Context(), key); err != nil {
		h.logger.Error().Err(err).Msg("Cache delete failed")
		h.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

		h.writeSuccess(w, http.StatusOK, map[string]interface{}{
			"key": key,
		})
}

// Helper methods

func (h *Handlers) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		h.logger.Error().Err(err).Msg("Failed to encode JSON response")
	}
}

func (h *Handlers) writeError(w http.ResponseWriter, status int, message string) {
	WriteError(w, status, message)
}

func (h *Handlers) writeSuccess(w http.ResponseWriter, status int, data interface{}) {
	WriteSuccess(w, status, data)
}

func (h *Handlers) getIntQuery(r *http.Request, key string, defaultValue int) int {
	value := r.URL.Query().Get(key)
	if value == "" {
		return defaultValue
	}
	
	var intValue int
	if _, err := fmt.Sscanf(value, "%d", &intValue); err != nil {
		return defaultValue
	}
	return intValue
}

