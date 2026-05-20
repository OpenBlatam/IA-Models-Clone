package api

import (
	"encoding/json"
	"net/http"
	"time"
)

// Response represents a standard API response
type Response struct {
	Status    string      `json:"status"`
	Data      interface{} `json:"data,omitempty"`
	Error     string      `json:"error,omitempty"`
	Timestamp string      `json:"timestamp"`
}

// SuccessResponse creates a success response
func SuccessResponse(data interface{}) *Response {
	return &Response{
		Status:    "success",
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
}

// ErrorResponse creates an error response
func ErrorResponse(message string) *Response {
	return &Response{
		Status:    "error",
		Error:     message,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
}

// WriteSuccess writes a success response
func WriteSuccess(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(SuccessResponse(data))
}

// WriteError writes an error response
func WriteError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(ErrorResponse(message))
}












