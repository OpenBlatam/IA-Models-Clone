package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Validator provides request validation
type Validator struct{}

// NewValidator creates a new validator
func NewValidator() *Validator {
	return &Validator{}
}

// ValidateGitCloneRequest validates Git clone request
func (v *Validator) ValidateGitCloneRequest(r *http.Request) error {
	var url, path string
	
	// Try JSON body first
	if r.Header.Get("Content-Type") == "application/json" {
		var req struct {
			URL  string `json:"url"`
			Path string `json:"path"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
			url = req.URL
			path = req.Path
		}
	}
	
	// Fallback to query parameters
	if url == "" {
		url = r.URL.Query().Get("url")
	}
	if path == "" {
		path = r.URL.Query().Get("path")
	}
	
	if url == "" {
		return fmt.Errorf("missing required parameter: url")
	}
	if path == "" {
		return fmt.Errorf("missing required parameter: path")
	}
	
	// Validate URL format
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return fmt.Errorf("invalid URL format")
	}
	
	return nil
}

// ValidateSearchRequest validates search request
func (v *Validator) ValidateSearchRequest(r *http.Request) error {
	query := r.URL.Query().Get("q")
	if query == "" {
		return fmt.Errorf("missing required parameter: q")
	}
	
	if len(query) < 2 {
		return fmt.Errorf("query must be at least 2 characters")
	}
	
	if len(query) > 500 {
		return fmt.Errorf("query too long (max 500 characters)")
	}
	
	return nil
}

// ValidateCacheRequest validates cache request
func (v *Validator) ValidateCacheRequest(r *http.Request) error {
	key := r.URL.Query().Get("key")
	if key == "" {
		return fmt.Errorf("missing required parameter: key")
	}
	
	if len(key) > 500 {
		return fmt.Errorf("key too long (max 500 characters)")
	}
	
	return nil
}

