package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/blatam-academy/social-video-transcriber/go_services/internal/config"
	"github.com/blatam-academy/social-video-transcriber/go_services/internal/openrouter"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/go-chi/httprate"
	"github.com/rs/zerolog/log"
)

type Handler struct {
	config   *config.Config
	orClient *openrouter.Client
}

func NewRouter(cfg *config.Config, orClient *openrouter.Client) *chi.Mux {
	h := &Handler{
		config:   cfg,
		orClient: orClient,
	}

	r := chi.NewRouter()

	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(time.Duration(cfg.RequestTimeout) * time.Second))

	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   cfg.CORSOrigins,
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-API-Key"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	r.Use(httprate.LimitByIP(cfg.RateLimitRPM, time.Minute))

	r.Get("/health", h.healthCheck)
	r.Get("/stats", h.getStats)

	r.Route("/api/v1", func(r chi.Router) {
		r.Route("/ai", func(r chi.Router) {
			r.Post("/analyze", h.analyzeContent)
			r.Post("/variants", h.generateVariants)
			r.Post("/summarize", h.summarize)
			r.Post("/keywords", h.extractKeywords)
			r.Post("/translate", h.translate)
			r.Post("/chat", h.chat)
		})

		r.Route("/batch", func(r chi.Router) {
			r.Post("/analyze", h.batchAnalyze)
			r.Post("/variants", h.batchVariants)
		})
	})

	return r
}

func (h *Handler) healthCheck(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":    "healthy",
		"service":   "go-transcriber-ai",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}

func (h *Handler) getStats(w http.ResponseWriter, r *http.Request) {
	stats := h.orClient.GetStats()
	stats["environment"] = h.config.Environment
	respondJSON(w, http.StatusOK, stats)
}

type AnalyzeRequest struct {
	Text string `json:"text"`
}

func (h *Handler) analyzeContent(w http.ResponseWriter, r *http.Request) {
	var req AnalyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.Text == "" {
		respondError(w, http.StatusBadRequest, "Text is required")
		return
	}

	result, err := h.orClient.AnalyzeContent(r.Context(), req.Text)
	if err != nil {
		log.Error().Err(err).Msg("Failed to analyze content")
		respondError(w, http.StatusInternalServerError, "Analysis failed")
		return
	}

	respondJSON(w, http.StatusOK, result)
}

type VariantsRequest struct {
	Text              string `json:"text"`
	Count             int    `json:"count"`
	PreserveFramework bool   `json:"preserve_framework"`
}

func (h *Handler) generateVariants(w http.ResponseWriter, r *http.Request) {
	var req VariantsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.Text == "" {
		respondError(w, http.StatusBadRequest, "Text is required")
		return
	}

	if req.Count < 1 {
		req.Count = 3
	}
	if req.Count > 10 {
		req.Count = 10
	}

	result, err := h.orClient.GenerateVariants(r.Context(), req.Text, req.Count, req.PreserveFramework)
	if err != nil {
		log.Error().Err(err).Msg("Failed to generate variants")
		respondError(w, http.StatusInternalServerError, "Variant generation failed")
		return
	}

	respondJSON(w, http.StatusOK, result)
}

type SummarizeRequest struct {
	Text  string `json:"text"`
	Style string `json:"style"`
}

func (h *Handler) summarize(w http.ResponseWriter, r *http.Request) {
	var req SummarizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.Text == "" {
		respondError(w, http.StatusBadRequest, "Text is required")
		return
	}

	summary, err := h.orClient.Summarize(r.Context(), req.Text, req.Style)
	if err != nil {
		log.Error().Err(err).Msg("Failed to summarize")
		respondError(w, http.StatusInternalServerError, "Summarization failed")
		return
	}

	respondJSON(w, http.StatusOK, map[string]string{
		"summary": summary,
		"style":   req.Style,
	})
}

type KeywordsRequest struct {
	Text        string `json:"text"`
	MaxKeywords int    `json:"max_keywords"`
}

func (h *Handler) extractKeywords(w http.ResponseWriter, r *http.Request) {
	var req KeywordsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.Text == "" {
		respondError(w, http.StatusBadRequest, "Text is required")
		return
	}

	if req.MaxKeywords < 1 {
		req.MaxKeywords = 10
	}

	keywords, err := h.orClient.ExtractKeywords(r.Context(), req.Text, req.MaxKeywords)
	if err != nil {
		log.Error().Err(err).Msg("Failed to extract keywords")
		respondError(w, http.StatusInternalServerError, "Keyword extraction failed")
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"keywords": keywords,
		"count":    len(keywords),
	})
}

type TranslateRequest struct {
	Text       string `json:"text"`
	TargetLang string `json:"target_lang"`
}

func (h *Handler) translate(w http.ResponseWriter, r *http.Request) {
	var req TranslateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.Text == "" || req.TargetLang == "" {
		respondError(w, http.StatusBadRequest, "Text and target_lang are required")
		return
	}

	translated, err := h.orClient.Translate(r.Context(), req.Text, req.TargetLang)
	if err != nil {
		log.Error().Err(err).Msg("Failed to translate")
		respondError(w, http.StatusInternalServerError, "Translation failed")
		return
	}

	respondJSON(w, http.StatusOK, map[string]string{
		"original":    req.Text,
		"translated":  translated,
		"target_lang": req.TargetLang,
	})
}

type ChatRequest struct {
	Messages []openrouter.Message `json:"messages"`
	Model    string               `json:"model"`
}

func (h *Handler) chat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if len(req.Messages) == 0 {
		respondError(w, http.StatusBadRequest, "Messages are required")
		return
	}

	chatReq := openrouter.ChatRequest{
		Model:    req.Model,
		Messages: req.Messages,
	}

	resp, err := h.orClient.Chat(r.Context(), chatReq)
	if err != nil {
		log.Error().Err(err).Msg("Failed to chat")
		respondError(w, http.StatusInternalServerError, "Chat failed")
		return
	}

	respondJSON(w, http.StatusOK, resp)
}

type BatchAnalyzeRequest struct {
	Texts []string `json:"texts"`
}

func (h *Handler) batchAnalyze(w http.ResponseWriter, r *http.Request) {
	var req BatchAnalyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if len(req.Texts) == 0 {
		respondError(w, http.StatusBadRequest, "Texts array is required")
		return
	}

	if len(req.Texts) > 10 {
		respondError(w, http.StatusBadRequest, "Maximum 10 texts per batch")
		return
	}

	results := make([]*openrouter.AnalysisResult, len(req.Texts))
	errors := make([]string, len(req.Texts))

	for i, text := range req.Texts {
		result, err := h.orClient.AnalyzeContent(r.Context(), text)
		if err != nil {
			errors[i] = err.Error()
		} else {
			results[i] = result
		}
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"errors":  errors,
		"total":   len(req.Texts),
	})
}

type BatchVariantsRequest struct {
	Texts []string `json:"texts"`
	Count int      `json:"count"`
}

func (h *Handler) batchVariants(w http.ResponseWriter, r *http.Request) {
	var req BatchVariantsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if len(req.Texts) == 0 {
		respondError(w, http.StatusBadRequest, "Texts array is required")
		return
	}

	if len(req.Texts) > 5 {
		respondError(w, http.StatusBadRequest, "Maximum 5 texts per batch")
		return
	}

	if req.Count < 1 {
		req.Count = 3
	}

	results := make([]*openrouter.VariantResult, len(req.Texts))
	errors := make([]string, len(req.Texts))

	for i, text := range req.Texts {
		result, err := h.orClient.GenerateVariants(r.Context(), text, req.Count, true)
		if err != nil {
			errors[i] = err.Error()
		} else {
			results[i] = result
		}
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"errors":  errors,
		"total":   len(req.Texts),
	})
}

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, status int, message string) {
	respondJSON(w, status, map[string]string{
		"error": message,
	})
}












