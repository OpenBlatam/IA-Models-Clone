package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/batch"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/cache"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/git"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/queue"
	"github.com/blatam-academy/github-autonomous-agent/go_services/internal/search"
)

// Advanced usage examples showing real-world patterns
func main() {
	fmt.Println("=== Advanced Usage Examples ===\n")

	// Example 1: Parallel repository processing with cache
	exampleParallelRepoProcessing()

	// Example 2: Search with indexing pipeline
	exampleSearchPipeline()

	// Example 3: High-throughput task processing
	exampleHighThroughputQueue()

	// Example 4: Batch processing with error handling
	exampleBatchWithRetry()
}

func exampleParallelRepoProcessing() {
	fmt.Println("1. Parallel Repository Processing with Cache")
	fmt.Println("-------------------------------------------")

	// Setup cache
	cacheService, err := cache.NewMultiTierCache(cache.Config{
		MemorySize:   10000,
		MemoryTTL:    10 * time.Minute,
		BadgerPath:   "/tmp/repo_cache",
		EnableBadger: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cacheService.Close()

	ctx := context.Background()
	repoURLs := []string{
		"https://github.com/user/repo1.git",
		"https://github.com/user/repo2.git",
		"https://github.com/user/repo3.git",
	}

	// Process repositories in parallel with caching
	for _, url := range repoURLs {
		cacheKey := fmt.Sprintf("repo:%s", url)

		// Check cache first
		if cached, found := cacheService.Get(ctx, cacheKey); found {
			fmt.Printf("  Cache hit for %s: %v\n", url, cached)
			continue
		}

		// Process repository (simulated)
		fmt.Printf("  Processing %s...\n", url)
		
		// In real usage: git.Clone(url, path, nil)
		// For now, simulate processing
		time.Sleep(100 * time.Millisecond)

		// Cache result
		cacheService.Set(ctx, cacheKey, "processed", 10*time.Minute)
	}

	fmt.Println()
}

func exampleSearchPipeline() {
	fmt.Println("2. Search with Indexing Pipeline")
	fmt.Println("---------------------------------")

	// Create search index
	index, err := search.NewIndex("/tmp/search_pipeline")
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	// Index documents in batch
	docs := []search.Document{
		{
			ID:   "doc-1",
			Text: "GitHub autonomous agent for repository management",
			Metadata: map[string]interface{}{
				"repo": "user/repo1",
				"type": "code",
			},
		},
		{
			ID:   "doc-2",
			Text: "High-performance Go services for Git operations",
			Metadata: map[string]interface{}{
				"repo": "user/repo2",
				"type": "documentation",
			},
		},
	}

	// Batch index
	err = index.IndexBatch(docs)
	if err != nil {
		log.Fatal(err)
	}

	// Search with facets
	results, err := index.Search("autonomous agent", search.SearchOptions{
		Limit:  10,
		Facets: []string{"type", "repo"},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("  Found %d results\n", results.Total)
	for _, hit := range results.Hits {
		fmt.Printf("  - %s (score: %.2f)\n", hit.ID, hit.Score)
	}

	fmt.Println()
}

func exampleHighThroughputQueue() {
	fmt.Println("3. High-Throughput Task Processing")
	fmt.Println("-----------------------------------")

	// Create queue with high concurrency
	queue, err := queue.NewTaskQueue(queue.Config{
		MaxWorkers: 100,
		QueueSize:  10000,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer queue.Stop()

	// Start processing
	queue.Start(func(task queue.Task) queue.TaskResult {
		// Simulate work
		time.Sleep(10 * time.Millisecond)
		return queue.TaskResult{
			TaskID:  task.ID,
			Success: true,
		}
	})

	// Enqueue many tasks
	start := time.Now()
	for i := 0; i < 1000; i++ {
		queue.Enqueue(queue.Task{
			ID:   fmt.Sprintf("task-%d", i),
			Type: "process",
			Data: fmt.Sprintf("data-%d", i),
		})
	}

	// Wait for processing
	time.Sleep(2 * time.Second)

	stats := queue.GetStats()
	duration := time.Since(start)

	fmt.Printf("  Processed %d tasks in %v\n", stats.Processed, duration)
	fmt.Printf("  Throughput: %.0f tasks/sec\n", float64(stats.Processed)/duration.Seconds())
	fmt.Println()
}

func exampleBatchWithRetry() {
	fmt.Println("4. Batch Processing with Error Handling")
	fmt.Println("----------------------------------------")

	processor, err := batch.NewProcessor(batch.Config{
		MaxWorkers: 10,
		BatchSize:  50,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer processor.Close()

	items := make([]batch.Item, 100)
	for i := 0; i < 100; i++ {
		items[i] = batch.Item{
			ID:   fmt.Sprintf("item-%d", i),
			Data: i,
		}
	}

	// Process with retry logic
	results := processor.Process(context.Background(), items, func(item batch.Item) error {
		// Simulate occasional failures
		if item.Data.(int)%10 == 0 {
			return fmt.Errorf("simulated error for item %s", item.ID)
		}
		return nil
	})

	// Count successes and failures
	successCount := 0
	failureCount := 0
	for _, result := range results {
		if result.Success {
			successCount++
		} else {
			failureCount++
		}
	}

	fmt.Printf("  Success: %d, Failed: %d\n", successCount, failureCount)
	fmt.Println()
}












