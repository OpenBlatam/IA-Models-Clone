import Config

# Production-specific configuration

config :faceless_video, FacelessVideoWeb.Endpoint,
  cache_static_manifest: "priv/static/cache_manifest.json",
  force_ssl: [hsts: true]

# Production logger - only warnings and above
config :logger, level: :warning

# Production Oban - enable all plugins
config :faceless_video, Oban,
  plugins: [
    {Oban.Plugins.Pruner, max_age: 60 * 60 * 24 * 7},
    {Oban.Plugins.Stager, interval: 1000},
    {Oban.Plugins.Lifeline, rescue_after: :timer.minutes(30)},
    {Oban.Plugins.Reindexer, schedule: "@weekly"},
    {Oban.Plugins.Cron,
     crontab: [
       {"0 3 * * *", FacelessVideo.Workers.Cleanup},
       {"0 * * * *", FacelessVideo.Workers.MetricsAggregator},
       {"*/5 * * * *", FacelessVideo.Workers.HealthCheck}
     ]}
  ]

# Production cluster topology (Kubernetes)
config :libcluster,
  topologies: [
    k8s: [
      strategy: Cluster.Strategy.Kubernetes.DNS,
      config: [
        service: "faceless-video-headless",
        application_name: "faceless_video",
        polling_interval: 10_000
      ]
    ]
  ]

# Enable distributed mode in production
config :faceless_video, :distributed, true

# Runtime configuration is loaded in runtime.exs
