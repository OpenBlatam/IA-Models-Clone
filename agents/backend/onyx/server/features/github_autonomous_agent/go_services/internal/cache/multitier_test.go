package cache

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestMultiTierCache(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "cache-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cache, err := NewMultiTierCache(Config{
		MemorySize:    1000,
		MemoryTTL:     5 * time.Minute,
		BadgerPath:    tmpDir,
		EnableBadger:  true,
		EnableRedis:   false,
	})
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()

	// Test Set and Get
	err = cache.Set(ctx, "test_key", "test_value", 5*time.Minute)
	if err != nil {
		t.Fatalf("Failed to set cache: %v", err)
	}

	val, found := cache.Get(ctx, "test_key")
	if !found {
		t.Fatal("Cache get failed - key not found")
	}

	if val != "test_value" {
		t.Fatalf("Expected 'test_value', got %v", val)
	}

	// Test Delete
	err = cache.Delete(ctx, "test_key")
	if err != nil {
		t.Fatalf("Failed to delete cache: %v", err)
	}

	_, found = cache.Get(ctx, "test_key")
	if found {
		t.Fatal("Cache delete failed - key still exists")
	}
}

func TestCacheStats(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "cache-stats-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cache, err := NewMultiTierCache(Config{
		MemorySize:   1000,
		MemoryTTL:    5 * time.Minute,
		BadgerPath:   tmpDir,
		EnableBadger: true,
		EnableRedis:  false,
	})
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	stats := cache.GetStats()
	if stats == nil {
		t.Fatal("GetStats returned nil")
	}
}












