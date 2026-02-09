import Config

# Test-specific configuration

config :faceless_video, FacelessVideoWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "test_secret_key_base_that_is_at_least_64_bytes_long_for_test_env",
  server: false

# Test database
config :faceless_video, FacelessVideo.Repo,
  database: "faceless_video_test#{System.get_env("MIX_TEST_PARTITION")}",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: System.schedulers_online() * 2

# Test pipeline (use dummy producer)
config :faceless_video, FacelessVideo.Pipeline.VideoProcessor,
  producer: :dummy

# Disable Oban in tests
config :faceless_video, Oban, testing: :inline

# Disable distributed mode in tests
config :faceless_video, :distributed, false

# Test logger - print only warnings
config :logger, level: :warning

# Initialize plugs at runtime for faster compilation
config :phoenix, :plug_init_mode, :runtime
