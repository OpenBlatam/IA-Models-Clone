import Config

# Development-specific configuration

config :faceless_video, FacelessVideoWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4000],
  check_origin: false,
  code_reloader: true,
  debug_errors: true,
  secret_key_base: "dev_secret_key_base_that_is_at_least_64_bytes_long_for_security",
  watchers: []

# Development database
config :faceless_video, FacelessVideo.Repo,
  database: "faceless_video_dev",
  stacktrace: true,
  show_sensitive_data_on_connection_error: true

# Development pipeline (use dummy producer)
config :faceless_video, FacelessVideo.Pipeline.VideoProcessor,
  producer: :dummy

# Disable distributed mode in dev
config :faceless_video, :distributed, false

# Development logger level
config :logger, :console, format: "[$level] $message\n"
config :logger, level: :debug

# Phoenix development settings
config :phoenix, :plug_init_mode, :runtime
config :phoenix, :stacktrace_depth, 20

# Disable Swoosh emails in development
config :faceless_video, FacelessVideo.Mailer, adapter: Swoosh.Adapters.Local
