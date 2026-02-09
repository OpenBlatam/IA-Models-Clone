/*
 * API Server - REST API for benchmark management
 * 
 * Refactored with:
 * - Better error handling
 * - Request validation
 * - Response formatting
 * - Metrics endpoint
 */

package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
	
	"github.com/blatam-academy/universal-model-benchmark-ai/workers"
)

type Server struct {
	scheduler *workers.Scheduler
	logger    *zap.Logger
}

type ErrorResponse struct {
	Error     string `json:"error"`
	Message   string `json:"message,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

type SuccessResponse struct {
	Data      interface{} `json:"data"`
	Timestamp int64       `json:"timestamp"`
}

func NewServer(scheduler *workers.Scheduler, logger *zap.Logger) *Server {
	return &Server{
		scheduler: scheduler,
		logger:    logger,
	}
}

func (s *Server) SetupRoutes() *gin.Engine {
	router := gin.Default()
	
	// Middleware
	router.Use(s.loggingMiddleware())
	router.Use(s.errorHandlingMiddleware())
	
	// Health check
	router.GET("/health", s.healthCheck)
	router.GET("/metrics", s.getMetrics)
	
	// Task management
	v1 := router.Group("/api/v1")
	{
		v1.POST("/tasks", s.createTask)
		v1.GET("/tasks", s.listTasks)
		v1.GET("/tasks/:id", s.getTask)
		v1.DELETE("/tasks/:id", s.deleteTask)
		v1.POST("/tasks/:id/retry", s.retryTask)
	}
	
	// Results
	v1.GET("/results", s.getResults)
	v1.GET("/results/:id", s.getResult)
	
	return router
}

func (s *Server) loggingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		
		c.Next()
		
		latency := time.Since(start)
		s.logger.Info("Request",
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.Int("status", c.Writer.Status()),
			zap.Duration("latency", latency),
		)
	}
}

func (s *Server) errorHandlingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()
		
		if len(c.Errors) > 0 {
			err := c.Errors.Last()
			s.logger.Error("Request error", zap.Error(err))
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error:     "internal_server_error",
				Message:   err.Error(),
				Timestamp: time.Now().Unix(),
			})
		}
	}
}

func (s *Server) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, SuccessResponse{
		Data: gin.H{
			"status":    "healthy",
			"timestamp": time.Now().Unix(),
		},
		Timestamp: time.Now().Unix(),
	})
}

func (s *Server) getMetrics(c *gin.Context) {
	stats := s.scheduler.GetStats()
	c.JSON(http.StatusOK, SuccessResponse{
		Data:      stats,
		Timestamp: time.Now().Unix(),
	})
}

func (s *Server) createTask(c *gin.Context) {
	var task workers.BenchmarkTask
	if err := c.ShouldBindJSON(&task); err != nil {
		s.respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	
	// Validate task
	if task.ID == "" {
		s.respondError(c, http.StatusBadRequest, "invalid_request", "task ID is required")
		return
	}
	
	if task.ModelName == "" {
		s.respondError(c, http.StatusBadRequest, "invalid_request", "model name is required")
		return
	}
	
	if task.Benchmark == "" {
		s.respondError(c, http.StatusBadRequest, "invalid_request", "benchmark is required")
		return
	}
	
	if err := s.scheduler.ScheduleTask(&task); err != nil {
		s.respondError(c, http.StatusInternalServerError, "scheduling_failed", err.Error())
		return
	}
	
	s.respondSuccess(c, http.StatusCreated, task)
}

func (s *Server) listTasks(c *gin.Context) {
	tasks := s.scheduler.GetAllTasks()
	s.respondSuccess(c, http.StatusOK, tasks)
}

func (s *Server) getTask(c *gin.Context) {
	taskID := c.Param("id")
	task, err := s.scheduler.GetTaskStatus(taskID)
	if err != nil {
		s.respondError(c, http.StatusNotFound, "task_not_found", err.Error())
		return
	}
	
	s.respondSuccess(c, http.StatusOK, task)
}

func (s *Server) deleteTask(c *gin.Context) {
	// Implementation for task deletion
	taskID := c.Param("id")
	s.logger.Info("Task deletion requested", zap.String("task_id", taskID))
	
	s.respondSuccess(c, http.StatusOK, gin.H{
		"message": "Task deletion not yet implemented",
		"task_id": taskID,
	})
}

func (s *Server) retryTask(c *gin.Context) {
	taskID := c.Param("id")
	task, err := s.scheduler.GetTaskStatus(taskID)
	if err != nil {
		s.respondError(c, http.StatusNotFound, "task_not_found", err.Error())
		return
	}
	
	// Reset task for retry
	task.Status = "queued"
	task.Error = ""
	task.Retries = 0
	
	if err := s.scheduler.ScheduleTask(task); err != nil {
		s.respondError(c, http.StatusInternalServerError, "retry_failed", err.Error())
		return
	}
	
	s.respondSuccess(c, http.StatusOK, task)
}

func (s *Server) getResults(c *gin.Context) {
	tasks := s.scheduler.GetAllTasks()
	results := make([]workers.BenchmarkTask, 0)
	
	for _, task := range tasks {
		if task.Status == "completed" && task.Result != nil {
			results = append(results, *task)
		}
	}
	
	s.respondSuccess(c, http.StatusOK, results)
}

func (s *Server) getResult(c *gin.Context) {
	taskID := c.Param("id")
	task, err := s.scheduler.GetTaskStatus(taskID)
	if err != nil {
		s.respondError(c, http.StatusNotFound, "task_not_found", err.Error())
		return
	}
	
	if task.Status != "completed" || task.Result == nil {
		s.respondError(c, http.StatusNotFound, "result_not_found", "Task not completed or result not available")
		return
	}
	
	s.respondSuccess(c, http.StatusOK, task.Result)
}

func (s *Server) respondSuccess(c *gin.Context, status int, data interface{}) {
	c.JSON(status, SuccessResponse{
		Data:      data,
		Timestamp: time.Now().Unix(),
	})
}

func (s *Server) respondError(c *gin.Context, status int, errorCode, message string) {
	c.JSON(status, ErrorResponse{
		Error:     errorCode,
		Message:   message,
		Timestamp: time.Now().Unix(),
	})
}
