package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds application configuration
type Config struct {
	Port         int
	LogLevel     string
	CacheConfig  CacheConfig
	SearchConfig SearchConfig
}

// CacheConfig holds cache configuration
type CacheConfig struct {
	MemorySize     int
	MemoryTTL      time.Duration
	BadgerPath     string
	EnableBadger   bool
	EnableRedis    bool
	RedisURL       string
}

// SearchConfig holds search configuration
type SearchConfig struct {
	IndexPath string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() *Config {
	port, _ := strconv.Atoi(getEnv("PORT", "8080"))
	
	return &Config{
		Port:     port,
		LogLevel: getEnv("LOG_LEVEL", "info"),
		CacheConfig: CacheConfig{
			MemorySize:   getIntEnv("CACHE_MEMORY_SIZE", 10000),
			MemoryTTL:   getDurationEnv("CACHE_MEMORY_TTL", 5*time.Minute),
			BadgerPath:   getEnv("CACHE_BADGER_PATH", "/tmp/agent_cache"),
			EnableBadger: getBoolEnv("CACHE_ENABLE_BADGER", true),
			EnableRedis:  getBoolEnv("CACHE_ENABLE_REDIS", false),
			RedisURL:     getEnv("REDIS_URL", ""),
		},
		SearchConfig: SearchConfig{
			IndexPath: getEnv("SEARCH_INDEX_PATH", "/tmp/agent_search"),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}












