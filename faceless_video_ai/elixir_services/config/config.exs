# General application configuration
import Config

config :faceless_video,
  ecto_repos: [FacelessVideo.Repo],
  generators: [timestamp_type: :utc_datetime],
  distributed: true

# Database configuration
config :faceless_video, FacelessVideo.Repo,
  database: "faceless_video_#{config_env()}",
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  pool_size: 10,
  show_sensitive_data_on_connection_error: true

# Phoenix Endpoint
config :faceless_video, FacelessVideoWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [json: FacelessVideoWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: FacelessVideo.PubSub,
  live_view: [signing_salt: "CHANGE_ME_IN_PROD"]

# Oban background jobs
config :faceless_video, Oban,
  repo: FacelessVideo.Repo,
  plugins: [
    {Oban.Plugins.Pruner, max_age: 60 * 60 * 24 * 7},
    {Oban.Plugins.Cron,
     crontab: [
       # Daily cleanup at 3 AM
       {"0 3 * * *", FacelessVideo.Workers.Cleanup},
       # Hourly metrics aggregation
       {"0 * * * *", FacelessVideo.Workers.MetricsAggregator}
     ]}
  ],
  queues: [
    default: 10,
    video_processing: [limit: 5],
    audio_processing: [limit: 10],
    notifications: [limit: 20],
    maintenance: [limit: 2]
  ]

# Broadway Pipeline
config :faceless_video, FacelessVideo.Pipeline.VideoProcessor,
  producer: :rabbitmq,
  processor_concurrency: 10,
  producer_concurrency: 2,
  batch_size: 5,
  batch_timeout: 30_000

# Cache configuration
config :faceless_video, FacelessVideo.Cache,
  model: :inclusive,
  levels: [
    {FacelessVideo.Cache.L1,
     gc_interval: :timer.hours(1),
     max_size: 100_000,
     allocated_memory: 100 * 1024 * 1024},
    {FacelessVideo.Cache.L2, []}
  ]

config :faceless_video, FacelessVideo.Cache.L1,
  backend: :shards,
  gc_interval: :timer.hours(1)

config :faceless_video, FacelessVideo.Cache.L2,
  conn_opts: [
    host: "localhost",
    port: 6379
  ]

# Rate limiting
config :faceless_video, :rate_limiter,
  enabled: true,
  backend: :redis

config :hammer,
  backend: {Hammer.Backend.Redis, [expiry_ms: 60_000 * 60, redix_config: []]}

# Cluster configuration
config :libcluster,
  topologies: []

# Logger
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :user_id, :job_id]

# OpenTelemetry
config :opentelemetry,
  span_processor: :batch,
  traces_exporter: :otlp

config :opentelemetry_exporter,
  otlp_protocol: :grpc,
  otlp_endpoint: "http://localhost:4317"

# JSON library
config :phoenix, :json_library, Jason

# Import environment specific config
import_config "#{config_env()}.exs"
