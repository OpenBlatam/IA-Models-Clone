// Package main provides the main entry point for the Go services API
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/blatam-academy/faceless-video-ai/go_services/internal/circuitbreaker"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/eventbus"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/health"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/openrouter"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/queue"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/ratelimit"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/webhook"
	"github.com/blatam-academy/faceless-video-ai/go_services/internal/websocket"
)

// Services holds all service instances
type Services struct {
	OpenRouter   *openrouter.Client
	WebSocket    *websocket.Manager
	Queue        *queue.Manager
	Webhook      *webhook.Service
	EventBus     *eventbus.Bus
	RateLimiter  *ratelimit.Limiter
	Health       *health.Checker
}

func main() {
	godotenv.Load()

	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	if os.Getenv("ENV") == "development" {
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	}

	services := initServices()
	router := setupRouter(services)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Info().Str("port", port).Msg("Starting Go services API")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("Server failed")
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info().Msg("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal().Err(err).Msg("Server forced to shutdown")
	}

	services.cleanup()
	log.Info().Msg("Server exited properly")
}

func initServices() *Services {
	openrouterClient := openrouter.NewClient(openrouter.ClientConfig{
		APIKey:  os.Getenv("OPENROUTER_API_KEY"),
		AppName: "FacelessVideoAI",
		AppURL:  os.Getenv("APP_URL"),
	})

	wsManager := websocket.NewManager()
	go wsManager.Run()

	queueManager := queue.NewManager(5)
	queueManager.Start()

	webhookService := webhook.NewService(webhook.DefaultConfig())

	eventBus := eventbus.NewBus(1000, 1000)

	rateLimiter := ratelimit.NewLimiter(ratelimit.DefaultConfig())

	healthChecker := health.NewChecker(30 * time.Second)
	healthChecker.Start()

	return &Services{
		OpenRouter:  openrouterClient,
		WebSocket:   wsManager,
		Queue:       queueManager,
		Webhook:     webhookService,
		EventBus:    eventBus,
		RateLimiter: rateLimiter,
		Health:      healthChecker,
	}
}

func (s *Services) cleanup() {
	s.Queue.Stop()
	s.Webhook.Close()
	s.EventBus.Close()
	s.Health.Stop()
}

func setupRouter(services *Services) *gin.Engine {
	if os.Getenv("ENV") == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(requestLogger())
	router.Use(corsMiddleware())

	router.GET("/health", gin.WrapF(services.Health.Handler()))
	router.GET("/health/live", gin.WrapF(services.Health.LivenessHandler()))
	router.GET("/health/ready", gin.WrapF(services.Health.ReadinessHandler()))

	api := router.Group("/api/v1")
	{
		api.Use(rateLimitMiddleware(services.RateLimiter))

		api.POST("/llm/chat", chatHandler(services))
		api.POST("/llm/enhance-script", enhanceScriptHandler(services))
		api.POST("/llm/generate-prompt", generatePromptHandler(services))
		api.GET("/llm/models", modelsHandler(services))

		api.POST("/webhooks/register", registerWebhookHandler(services))
		api.DELETE("/webhooks/:video_id", unregisterWebhookHandler(services))

		api.POST("/jobs/enqueue", enqueueJobHandler(services))
		api.GET("/jobs/:job_id", getJobHandler(services))
		api.DELETE("/jobs/:job_id", cancelJobHandler(services))
		api.GET("/jobs/stats", jobStatsHandler(services))

		api.POST("/events/publish", publishEventHandler(services))
		api.GET("/events/history", eventHistoryHandler(services))

		api.GET("/stats", statsHandler(services))
		api.GET("/circuit-breakers", circuitBreakerHandler(services))
	}

	router.GET("/ws/:video_id", wsHandler(services))

	return router
}

func requestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path

		c.Next()

		log.Info().
			Str("method", c.Request.Method).
			Str("path", path).
			Int("status", c.Writer.Status()).
			Dur("latency", time.Since(start)).
			Str("ip", c.ClientIP()).
			Msg("Request")
	}
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func rateLimitMiddleware(limiter *ratelimit.Limiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := c.ClientIP()
		result := limiter.Allow(key, 100, 200)

		c.Header("X-RateLimit-Limit", "200")
		c.Header("X-RateLimit-Remaining", string(rune(result.Remaining)))

		if !result.Allowed {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": "Rate limit exceeded",
				"retry_after": result.ResetAt - time.Now().Unix(),
			})
			return
		}

		c.Next()
	}
}

func chatHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openrouter.ChatRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		breaker := circuitbreaker.GetBreaker("openrouter")
		result, err := breaker.Execute(func() (interface{}, error) {
			return services.OpenRouter.Chat(c.Request.Context(), &req)
		})

		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, result)
	}
}

func enhanceScriptHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			Script   string              `json:"script" binding:"required"`
			Language string              `json:"language"`
			Model    openrouter.Model    `json:"model"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if req.Language == "" {
			req.Language = "es"
		}

		enhanced, err := services.OpenRouter.EnhanceScript(
			c.Request.Context(),
			req.Script,
			req.Language,
			req.Model,
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"original": req.Script,
			"enhanced": enhanced,
		})
	}
}

func generatePromptHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			Text  string           `json:"text" binding:"required"`
			Style string           `json:"style"`
			Model openrouter.Model `json:"model"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if req.Style == "" {
			req.Style = "realistic"
		}

		prompt, err := services.OpenRouter.GenerateImagePrompt(
			c.Request.Context(),
			req.Text,
			req.Style,
			req.Model,
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"prompt": prompt})
	}
}

func modelsHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		models, err := services.OpenRouter.GetModels(c.Request.Context())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"models": models})
	}
}

func registerWebhookHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			VideoID string `json:"video_id" binding:"required"`
			URL     string `json:"url" binding:"required"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		import "github.com/google/uuid"
		videoID, err := uuid.Parse(req.VideoID)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid video ID"})
			return
		}

		services.Webhook.Register(videoID, req.URL)
		c.JSON(http.StatusOK, gin.H{"message": "Webhook registered"})
	}
}

func unregisterWebhookHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		import "github.com/google/uuid"
		videoID, err := uuid.Parse(c.Param("video_id"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid video ID"})
			return
		}

		services.Webhook.UnregisterAll(videoID)
		c.JSON(http.StatusOK, gin.H{"message": "Webhooks unregistered"})
	}
}

func enqueueJobHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			Type     string                 `json:"type" binding:"required"`
			Priority int                    `json:"priority"`
			Data     map[string]interface{} `json:"data"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		jobID, err := services.Queue.EnqueueWithData(
			req.Type,
			queue.Priority(req.Priority),
			req.Data,
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"job_id": jobID.String()})
	}
}

func getJobHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		import "github.com/google/uuid"
		jobID, err := uuid.Parse(c.Param("job_id"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid job ID"})
			return
		}

		job := services.Queue.GetJob(jobID)
		if job == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Job not found"})
			return
		}

		c.JSON(http.StatusOK, job)
	}
}

func cancelJobHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		import "github.com/google/uuid"
		jobID, err := uuid.Parse(c.Param("job_id"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid job ID"})
			return
		}

		if err := services.Queue.CancelJob(jobID); err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"message": "Job cancelled"})
	}
}

func jobStatsHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, services.Queue.GetStats())
	}
}

func publishEventHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			Type string                 `json:"type" binding:"required"`
			Data map[string]interface{} `json:"data"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if err := services.EventBus.Publish(req.Type, req.Data); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"message": "Event published"})
	}
}

func eventHistoryHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		eventType := c.Query("type")
		limit := 100

		events := services.EventBus.GetHistory(eventType, limit)
		c.JSON(http.StatusOK, gin.H{"events": events})
	}
}

func statsHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"openrouter":       services.OpenRouter.GetStats(),
			"websocket":        services.WebSocket.GetStats(),
			"queue":            services.Queue.GetStats(),
			"webhook":          services.Webhook.GetStats(),
			"eventbus":         services.EventBus.GetStats(),
			"rate_limiter":     services.RateLimiter.GetStats(),
			"circuit_breakers": circuitbreaker.GetAllStats(),
		})
	}
}

func circuitBreakerHandler(services *Services) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, circuitbreaker.GetAllStats())
	}
}

func wsHandler(services *Services) gin.HandlerFunc {
	import "github.com/gorilla/websocket"
	var upgrader = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}

	return func(c *gin.Context) {
		import "github.com/google/uuid"
		videoID, err := uuid.Parse(c.Param("video_id"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid video ID"})
			return
		}

		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			log.Error().Err(err).Msg("Failed to upgrade WebSocket connection")
			return
		}

		client := services.WebSocket.RegisterClient(conn, videoID)
		go client.WritePump()
		go client.ReadPump()
	}
}




