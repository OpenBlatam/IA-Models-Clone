package config

import (
	"fmt"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	Port             string
	OpenRouterAPIKey string
	RedisURL         string
	CORSOrigins      []string
	RateLimitRPM     int
	MaxConcurrency   int
	RequestTimeout   int
	Environment      string
}

func Load() (*Config, error) {
	_ = godotenv.Load()

	port := getEnv("GO_PORT", "8081")
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENROUTER_API_KEY is required")
	}

	return &Config{
		Port:             port,
		OpenRouterAPIKey: apiKey,
		RedisURL:         getEnv("REDIS_URL", ""),
		CORSOrigins:      []string{"*"},
		RateLimitRPM:     getEnvInt("RATE_LIMIT_RPM", 60),
		MaxConcurrency:   getEnvInt("MAX_CONCURRENCY", 10),
		RequestTimeout:   getEnvInt("REQUEST_TIMEOUT", 60),
		Environment:      getEnv("ENVIRONMENT", "development"),
	}, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}












