import Config

# Runtime configuration - loaded at runtime, not compile time
# This is where we read environment variables

if config_env() == :prod do
  # Database URL
  database_url =
    System.get_env("DATABASE_URL") ||
      raise """
      environment variable DATABASE_URL is missing.
      For example: ecto://USER:PASS@HOST/DATABASE
      """

  maybe_ipv6 = if System.get_env("ECTO_IPV6") in ~w(true 1), do: [:inet6], else: []

  config :faceless_video, FacelessVideo.Repo,
    url: database_url,
    pool_size: String.to_integer(System.get_env("POOL_SIZE") || "10"),
    socket_options: maybe_ipv6,
    ssl: System.get_env("DATABASE_SSL") == "true"

  # Secret key base
  secret_key_base =
    System.get_env("SECRET_KEY_BASE") ||
      raise """
      environment variable SECRET_KEY_BASE is missing.
      You can generate one by calling: mix phx.gen.secret
      """

  host = System.get_env("PHX_HOST") || "example.com"
  port = String.to_integer(System.get_env("PORT") || "4000")

  config :faceless_video, FacelessVideoWeb.Endpoint,
    url: [host: host, port: 443, scheme: "https"],
    http: [
      ip: {0, 0, 0, 0, 0, 0, 0, 0},
      port: port
    ],
    secret_key_base: secret_key_base

  # Redis configuration
  redis_url = System.get_env("REDIS_URL") || "redis://localhost:6379"

  config :faceless_video, FacelessVideo.Cache.L2,
    conn_opts: [url: redis_url]

  config :hammer,
    backend:
      {Hammer.Backend.Redis,
       [
         expiry_ms: 60_000 * 60,
         redix_config: [url: redis_url]
       ]}

  # RabbitMQ configuration
  if System.get_env("RABBITMQ_URL") do
    config :faceless_video, FacelessVideo.Pipeline.VideoProcessor,
      producer: :rabbitmq,
      rabbitmq_url: System.get_env("RABBITMQ_URL")
  end

  # Kafka configuration
  if System.get_env("KAFKA_BROKERS") do
    brokers =
      System.get_env("KAFKA_BROKERS")
      |> String.split(",")
      |> Enum.map(fn broker ->
        [host, port] = String.split(broker, ":")
        {host, String.to_integer(port)}
      end)

    config :faceless_video, FacelessVideo.Pipeline.VideoProcessor,
      producer: :kafka,
      kafka_brokers: brokers
  end

  # OpenAI configuration
  if System.get_env("OPENAI_API_KEY") do
    config :faceless_video, :openai,
      api_key: System.get_env("OPENAI_API_KEY"),
      organization: System.get_env("OPENAI_ORG_ID")
  end

  # AWS S3 configuration
  if System.get_env("AWS_ACCESS_KEY_ID") do
    config :ex_aws,
      access_key_id: System.get_env("AWS_ACCESS_KEY_ID"),
      secret_access_key: System.get_env("AWS_SECRET_ACCESS_KEY"),
      region: System.get_env("AWS_REGION") || "us-east-1"

    config :faceless_video, :storage,
      bucket: System.get_env("S3_BUCKET"),
      cdn_url: System.get_env("CDN_URL")
  end

  # Sentry error tracking
  if System.get_env("SENTRY_DSN") do
    config :sentry,
      dsn: System.get_env("SENTRY_DSN"),
      environment_name: config_env(),
      enable_source_code_context: true,
      root_source_code_paths: [File.cwd!()]
  end

  # OpenTelemetry exporter
  if System.get_env("OTEL_EXPORTER_OTLP_ENDPOINT") do
    config :opentelemetry_exporter,
      otlp_endpoint: System.get_env("OTEL_EXPORTER_OTLP_ENDPOINT")
  end
end
