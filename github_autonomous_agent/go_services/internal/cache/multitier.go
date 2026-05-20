package cache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dgraph-io/badger/v4"
	"github.com/dgraph-io/ristretto"
	"github.com/patrickmn/go-cache"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog"
)

var log = zerolog.New(nil).With().Timestamp().Logger()

// MultiTierCache provides multi-tier caching with automatic fallback
type MultiTierCache struct {
	memory   *cache.Cache
	badger   *badger.DB
	ristretto *ristretto.Cache
	redis    *redis.Client
	logger   zerolog.Logger
	mu       sync.RWMutex
}

// Config for MultiTierCache
type Config struct {
	MemorySize     int           // Memory cache size
	MemoryTTL      time.Duration // Memory cache TTL
	BadgerPath     string        // BadgerDB path (optional)
	RistrettoSize  int64         // Ristretto cache size (optional)
	RedisURL       string        // Redis URL (optional)
	EnableBadger   bool
	EnableRistretto bool
	EnableRedis    bool
}

// NewMultiTierCache creates a new multi-tier cache
func NewMultiTierCache(cfg Config) (*MultiTierCache, error) {
	mtc := &MultiTierCache{
		memory: cache.New(cfg.MemoryTTL, cfg.MemoryTTL*2),
		logger: log.With().Str("component", "cache").Logger(),
	}

	// Initialize BadgerDB if enabled
	if cfg.EnableBadger && cfg.BadgerPath != "" {
		opts := badger.DefaultOptions(cfg.BadgerPath)
		opts.Logger = nil // Disable badger logging
		db, err := badger.Open(opts)
		if err != nil {
			return nil, fmt.Errorf("failed to open badger: %w", err)
		}
		mtc.badger = db
	}

	// Initialize Ristretto if enabled
	if cfg.EnableRistretto && cfg.RistrettoSize > 0 {
		ristrettoCache, err := ristretto.NewCache(&ristretto.Config{
			NumCounters: cfg.RistrettoSize * 10,
			MaxCost:     cfg.RistrettoSize,
			BufferItems: 64,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create ristretto cache: %w", err)
		}
		mtc.ristretto = ristrettoCache
	}

	// Initialize Redis if enabled
	if cfg.EnableRedis && cfg.RedisURL != "" {
		opt, err := redis.ParseURL(cfg.RedisURL)
		if err != nil {
			return nil, fmt.Errorf("failed to parse redis URL: %w", err)
		}
		mtc.redis = redis.NewClient(opt)
	}

	return mtc, nil
}

// Set sets a value in all enabled tiers
func (mtc *MultiTierCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	mtc.mu.Lock()
	defer mtc.mu.Unlock()

	// Set in memory cache (always enabled)
	mtc.memory.Set(key, value, ttl)

	// Set in Ristretto if enabled
	if mtc.ristretto != nil {
		mtc.ristretto.SetWithTTL(key, value, 1, ttl)
	}

	// Set in BadgerDB if enabled (persistent)
	if mtc.badger != nil {
		err := mtc.badger.Update(func(txn *badger.Txn) error {
			entry := badger.NewEntry([]byte(key), []byte(fmt.Sprintf("%v", value))).WithTTL(ttl)
			return txn.SetEntry(entry)
		})
		if err != nil {
			mtc.logger.Warn().Err(err).Str("key", key).Msg("failed to set in badger")
		}
	}

	// Set in Redis if enabled (distributed)
	if mtc.redis != nil {
		err := mtc.redis.Set(ctx, key, value, ttl).Err()
		if err != nil {
			mtc.logger.Warn().Err(err).Str("key", key).Msg("failed to set in redis")
		}
	}

	return nil
}

// Get gets a value with automatic fallback through tiers
func (mtc *MultiTierCache) Get(ctx context.Context, key string) (interface{}, bool) {
	mtc.mu.RLock()
	defer mtc.mu.RUnlock()

	// Try memory cache first (fastest)
	if val, found := mtc.memory.Get(key); found {
		return val, true
	}

	// Try Ristretto
	if mtc.ristretto != nil {
		if val, found := mtc.ristretto.Get(key); found {
			// Promote to memory cache
			mtc.memory.Set(key, val, 5*time.Minute)
			return val, true
		}
	}

	// Try BadgerDB
	if mtc.badger != nil {
		var val []byte
		err := mtc.badger.View(func(txn *badger.Txn) error {
			item, err := txn.Get([]byte(key))
			if err != nil {
				return err
			}
			return item.Value(func(v []byte) error {
				val = make([]byte, len(v))
				copy(val, v)
				return nil
			})
		})
		if err == nil {
			// Promote to memory cache
			mtc.memory.Set(key, string(val), 5*time.Minute)
			return string(val), true
		}
	}

	// Try Redis
	if mtc.redis != nil {
		val, err := mtc.redis.Get(ctx, key).Result()
		if err == nil {
			// Promote to memory cache
			mtc.memory.Set(key, val, 5*time.Minute)
			return val, true
		}
	}

	return nil, false
}

// Delete deletes a key from all tiers
func (mtc *MultiTierCache) Delete(ctx context.Context, key string) error {
	mtc.mu.Lock()
	defer mtc.mu.Unlock()

	mtc.memory.Delete(key)

	if mtc.ristretto != nil {
		mtc.ristretto.Del(key)
	}

	if mtc.badger != nil {
		err := mtc.badger.Update(func(txn *badger.Txn) error {
			return txn.Delete([]byte(key))
		})
		if err != nil {
			mtc.logger.Warn().Err(err).Str("key", key).Msg("failed to delete from badger")
		}
	}

	if mtc.redis != nil {
		err := mtc.redis.Del(ctx, key).Err()
		if err != nil {
			mtc.logger.Warn().Err(err).Str("key", key).Msg("failed to delete from redis")
		}
	}

	return nil
}

// Clear clears all caches
func (mtc *MultiTierCache) Clear() {
	mtc.mu.Lock()
	defer mtc.mu.Unlock()

	mtc.memory.Flush()

	if mtc.ristretto != nil {
		mtc.ristretto.Clear()
	}

	if mtc.badger != nil {
		mtc.badger.DropAll()
	}

	if mtc.redis != nil {
		mtc.redis.FlushDB(context.Background())
	}
}

// Close closes all cache connections
func (mtc *MultiTierCache) Close() error {
	if mtc.badger != nil {
		if err := mtc.badger.Close(); err != nil {
			return err
		}
	}

	if mtc.redis != nil {
		if err := mtc.redis.Close(); err != nil {
			return err
		}
	}

	return nil
}

// GetStats returns cache statistics
func (mtc *MultiTierCache) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})

	// Memory cache stats
	items := mtc.memory.Items()
	stats["memory_items"] = len(items)

	// Ristretto stats
	if mtc.ristretto != nil {
		metrics := mtc.ristretto.Metrics
		stats["ristretto_hits"] = metrics.Hits()
		stats["ristretto_misses"] = metrics.Misses()
		stats["ristretto_cost"] = metrics.CostAdded()
	}

	// Redis stats
	if mtc.redis != nil {
		info := mtc.redis.Info(context.Background(), "stats")
		stats["redis_connected"] = info.Err() == nil
	}

	return stats
}












