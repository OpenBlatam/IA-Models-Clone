package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// HTTP metrics
	HTTPRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	HTTPRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	// Cache metrics
	CacheHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cache_hits_total",
			Help: "Total number of cache hits",
		},
		[]string{"cache_type"},
	)

	CacheMisses = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cache_misses_total",
			Help: "Total number of cache misses",
		},
		[]string{"cache_type"},
	)

	CacheSize = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "cache_size",
			Help: "Current cache size",
		},
		[]string{"cache_type"},
	)

	// Git operations metrics
	GitOperationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "git_operations_total",
			Help: "Total number of Git operations",
		},
		[]string{"operation", "status"},
	)

	GitOperationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "git_operation_duration_seconds",
			Help:    "Git operation duration in seconds",
			Buckets: []float64{0.1, 0.5, 1.0, 2.0, 5.0, 10.0},
		},
		[]string{"operation"},
	)

	// Search metrics
	SearchQueriesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "search_queries_total",
			Help: "Total number of search queries",
		},
		[]string{"status"},
	)

	SearchQueryDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "search_query_duration_seconds",
			Help:    "Search query duration in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5},
		},
		[]string{},
	)

	// Queue metrics
	QueueSize = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "queue_size",
			Help: "Current queue size",
		},
		[]string{"queue_name"},
	)

	QueueProcessed = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "queue_processed_total",
			Help: "Total number of processed queue items",
		},
		[]string{"queue_name", "status"},
	)

	// Batch processing metrics
	BatchProcessed = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "batch_processed_total",
			Help: "Total number of processed batch items",
		},
		[]string{"status"},
	)

	BatchDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "batch_duration_seconds",
			Help:    "Batch processing duration in seconds",
			Buckets: []float64{0.1, 0.5, 1.0, 5.0, 10.0, 30.0},
		},
		[]string{},
	)
)












