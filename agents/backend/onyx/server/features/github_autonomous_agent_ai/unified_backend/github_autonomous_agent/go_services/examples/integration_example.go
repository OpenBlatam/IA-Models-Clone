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

func main() {
	// Example 1: Git Operations
	fmt.Println("=== Git Operations Example ===")
	repo, err := git.OpenRepository("/path/to/repo")
	if err != nil {
		log.Printf("Note: Repository not found, this is just an example")
	} else {
		commits, err := repo.GetCommits("main", 10)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Found %d commits\n", len(commits))
	}

	// Example 2: Multi-Tier Cache
	fmt.Println("\n=== Cache Example ===")
	cacheService, err := cache.NewMultiTierCache(cache.Config{
		MemorySize:    10000,
		MemoryTTL:     5 * time.Minute,
		BadgerPath:    "/tmp/example_cache",
		EnableBadger:  true,
		EnableRedis:   false,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cacheService.Close()

	ctx := context.Background()
	cacheService.Set(ctx, "example_key", "example_value", 5*time.Minute)
	val, found := cacheService.Get(ctx, "example_key")
	if found {
		fmt.Printf("Cache hit: %v\n", val)
	}

	// Example 3: Full-Text Search
	fmt.Println("\n=== Search Example ===")
	index, err := search.NewIndex("/tmp/example_search")
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	doc := search.Document{
		ID:   "doc-1",
		Text: "GitHub autonomous agent for repository management",
		Metadata: map[string]interface{}{
			"repo": "user/repo",
			"type": "code",
		},
	}
	index.Index(doc)

	results, err := index.Search("autonomous agent", search.SearchOptions{
		Limit: 10,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Search found %d results\n", results.Total)

	// Example 4: Task Queue
	fmt.Println("\n=== Task Queue Example ===")
	taskQueue, err := queue.NewTaskQueue(queue.Config{
		MaxWorkers: 10,
		QueueSize:  100,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer taskQueue.Stop()

	taskQueue.Start(func(task queue.Task) queue.TaskResult {
		fmt.Printf("Processing task: %s\n", task.ID)
		return queue.TaskResult{
			TaskID:  task.ID,
			Success: true,
		}
	})

	taskQueue.Enqueue(queue.Task{
		ID:   "task-1",
		Type: "example",
		Data: "example data",
	})

	time.Sleep(100 * time.Millisecond)
	stats := taskQueue.GetStats()
	fmt.Printf("Queue stats: Enqueued=%d, Processed=%d\n", stats.Enqueued, stats.Processed)

	// Example 5: Batch Processing
	fmt.Println("\n=== Batch Processing Example ===")
	processor, err := batch.NewProcessor(batch.Config{
		MaxWorkers: 10,
		BatchSize:  5,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer processor.Close()

	items := []batch.Item{
		{ID: "1", Data: "item 1"},
		{ID: "2", Data: "item 2"},
		{ID: "3", Data: "item 3"},
	}

	results := processor.Process(context.Background(), items, func(item batch.Item) error {
		fmt.Printf("Processing item: %s\n", item.ID)
		return nil
	})

	fmt.Printf("Processed %d items\n", len(results))
}












