package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidateGitCloneRequest(t *testing.T) {
	validator := NewValidator()

	tests := []struct {
		name    string
		request *http.Request
		wantErr bool
	}{
		{
			name:    "valid request with query params",
			request: httptest.NewRequest(http.MethodPost, "/api/v1/git/clone?url=https://github.com/user/repo.git&path=/tmp/repo", nil),
			wantErr: false,
		},
		{
			name:    "missing url",
			request: httptest.NewRequest(http.MethodPost, "/api/v1/git/clone?path=/tmp/repo", nil),
			wantErr: true,
		},
		{
			name:    "missing path",
			request: httptest.NewRequest(http.MethodPost, "/api/v1/git/clone?url=https://github.com/user/repo.git", nil),
			wantErr: true,
		},
		{
			name:    "invalid URL format",
			request: httptest.NewRequest(http.MethodPost, "/api/v1/git/clone?url=invalid&path=/tmp/repo", nil),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateGitCloneRequest(tt.request)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateSearchRequest(t *testing.T) {
	validator := NewValidator()

	tests := []struct {
		name    string
		request *http.Request
		wantErr bool
	}{
		{
			name:    "valid query",
			request: httptest.NewRequest(http.MethodGet, "/api/v1/search?q=test", nil),
			wantErr: false,
		},
		{
			name:    "missing query",
			request: httptest.NewRequest(http.MethodGet, "/api/v1/search", nil),
			wantErr: true,
		},
		{
			name:    "query too short",
			request: httptest.NewRequest(http.MethodGet, "/api/v1/search?q=a", nil),
			wantErr: true,
		},
		{
			name:    "query too long",
			request: httptest.NewRequest(http.MethodGet, "/api/v1/search?q="+string(make([]byte, 501)), nil),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateSearchRequest(tt.request)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}












