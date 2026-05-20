defmodule FacelessVideo.Application do
  @moduledoc """
  Main application module for FacelessVideo.

  Starts the supervision tree with configurable components:
  - Phoenix endpoint for web/API
  - Broadway pipelines for video/audio processing
  - Distributed job queue (Oban)
  - Cluster management (libcluster + Horde)
  - Distributed cache layer (Nebulex)
  - Event broadcasting system
  """
  use Application

  require Logger

  @impl true
  def start(_type, _args) do
    Logger.info("Starting FacelessVideo application...")

    children = build_children()

    opts = [
      strategy: :one_for_one,
      name: FacelessVideo.Supervisor,
      max_restarts: 10,
      max_seconds: 60
    ]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        log_startup_info()
        {:ok, pid}

      {:error, reason} = error ->
        Logger.error("Failed to start FacelessVideo: #{inspect(reason)}")
        error
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("Shutting down FacelessVideo application...")
    :ok
  end

  @impl true
  def config_change(changed, _new, removed) do
    FacelessVideoWeb.Endpoint.config_change(changed, removed)
    :ok
  end

  # ============================================
  # Child Spec Builders
  # ============================================

  defp build_children do
    core_children() ++
      distributed_children() ++
      pipeline_children() ++
      web_children()
  end

  defp core_children do
    [
      # Database repository
      FacelessVideo.Repo,

      # Telemetry supervisor
      FacelessVideoWeb.Telemetry,

      # PubSub system
      {Phoenix.PubSub, name: FacelessVideo.PubSub},

      # HTTP client pool
      {Finch,
       name: FacelessVideo.Finch,
       pools: %{
         :default => [size: 25, count: 4],
         "https://api.openai.com" => [size: 10, count: 2, protocol: :http2],
         "https://api.anthropic.com" => [size: 10, count: 2, protocol: :http2]
       }},

      # Distributed cache
      cache_spec(),

      # Rate limiter
      rate_limiter_spec()
    ]
    |> Enum.reject(&is_nil/1)
  end

  defp distributed_children do
    if distributed_enabled?() do
      [
        # Cluster supervisor
        cluster_supervisor(),

        # Distributed registry (Horde)
        {Horde.Registry,
         name: FacelessVideo.Registry,
         keys: :unique,
         members: :auto,
         delta_crdt_options: [sync_interval: 100]},

        # Distributed supervisor
        {Horde.DynamicSupervisor,
         name: FacelessVideo.DynamicSupervisor,
         strategy: :one_for_one,
         members: :auto,
         distribution_strategy: Horde.UniformQuorumDistribution}
      ]
    else
      Logger.info("Distributed mode disabled, using local supervision")
      []
    end
  end

  defp pipeline_children do
    pipelines =
      []
      |> maybe_add_pipeline(:video, FacelessVideo.Pipeline.VideoProcessor)
      |> maybe_add_pipeline(:audio, FacelessVideo.Pipeline.AudioProcessor)

    [
      # Background job processor
      {Oban, oban_config()},

      # Event broadcaster
      FacelessVideo.Events.Broadcaster
    ] ++ pipelines
  end

  defp web_children do
    [
      # Presence tracking
      FacelessVideoWeb.Presence,

      # Phoenix endpoint (must be last)
      FacelessVideoWeb.Endpoint
    ]
  end

  # ============================================
  # Spec Helpers
  # ============================================

  defp cluster_supervisor do
    topologies = Application.get_env(:libcluster, :topologies, [])

    if topologies == [] do
      Logger.warning("No cluster topologies configured")
      nil
    else
      {Cluster.Supervisor, [topologies, [name: FacelessVideo.ClusterSupervisor]]}
    end
  end

  defp cache_spec do
    case Application.get_env(:faceless_video, FacelessVideo.Cache) do
      nil ->
        Logger.warning("Cache not configured, using default settings")
        FacelessVideo.Cache

      _config ->
        FacelessVideo.Cache
    end
  end

  defp rate_limiter_spec do
    case Application.get_env(:faceless_video, :rate_limiter) do
      nil -> nil
      _config -> FacelessVideo.RateLimiter
    end
  end

  defp maybe_add_pipeline(pipelines, type, module) do
    if pipeline_enabled?(type) do
      [module | pipelines]
    else
      Logger.info("#{type} pipeline disabled")
      pipelines
    end
  end

  defp oban_config do
    Application.get_env(:faceless_video, Oban, [
      repo: FacelessVideo.Repo,
      plugins: [
        {Oban.Plugins.Pruner, max_age: 60 * 60 * 24 * 7},
        {Oban.Plugins.Cron, crontab: []}
      ],
      queues: [
        default: 10,
        video_processing: [limit: 5, paused: false],
        audio_processing: [limit: 10],
        notifications: [limit: 20]
      ]
    ])
  end

  # ============================================
  # Configuration Helpers
  # ============================================

  defp distributed_enabled? do
    Application.get_env(:faceless_video, :distributed, true)
  end

  defp pipeline_enabled?(type) do
    config = Application.get_env(:faceless_video, :pipelines, %{})
    Map.get(config, type, true)
  end

  # ============================================
  # Startup Logging
  # ============================================

  defp log_startup_info do
    Logger.info("""

    ╔═══════════════════════════════════════════════════════════╗
    ║            FacelessVideo Started Successfully             ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Version:      #{Application.spec(:faceless_video, :vsn) |> to_string() |> String.pad_trailing(41)}║
    ║  Environment:  #{Mix.env() |> to_string() |> String.pad_trailing(41)}║
    ║  Node:         #{Node.self() |> to_string() |> String.slice(0, 40) |> String.pad_trailing(41)}║
    ║  Distributed:  #{distributed_enabled?() |> to_string() |> String.pad_trailing(41)}║
    ╚═══════════════════════════════════════════════════════════╝
    """)
  end
end
