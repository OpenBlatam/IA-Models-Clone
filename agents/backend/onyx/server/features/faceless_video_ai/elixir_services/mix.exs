defmodule FacelessVideo.MixProject do
  @moduledoc """
  Mix project configuration for FacelessVideo - a distributed video processing pipeline.
  """
  use Mix.Project

  @version "1.0.0"
  @source_url "https://github.com/blatam-academy/faceless-video"

  def project do
    [
      app: :faceless_video,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      consolidate_protocols: Mix.env() != :dev,
      aliases: aliases(),
      deps: deps(),
      releases: releases(),

      # Project metadata
      name: "FacelessVideo",
      description: description(),
      source_url: @source_url,
      homepage_url: @source_url,
      package: package(),

      # Documentation
      docs: docs(),

      # Dialyzer
      dialyzer: dialyzer(),

      # Test coverage
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: preferred_cli_env()
    ]
  end

  def application do
    [
      mod: {FacelessVideo.Application, []},
      extra_applications: extra_applications(Mix.env())
    ]
  end

  defp extra_applications(:dev), do: [:logger, :runtime_tools, :os_mon, :observer]
  defp extra_applications(:test), do: [:logger, :runtime_tools]
  defp extra_applications(_), do: [:logger, :runtime_tools, :os_mon]

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp description do
    """
    Distributed video processing pipeline with fault-tolerance, built on Broadway and OTP.
    Features AI-powered video generation, real-time streaming, and cluster support.
    """
  end

  defp deps do
    [
      # ============================================
      # Core Framework
      # ============================================
      {:phoenix, "~> 1.7.10"},
      {:phoenix_html, "~> 4.0"},
      {:phoenix_live_view, "~> 0.20.1"},
      {:phoenix_live_dashboard, "~> 0.8.2"},
      {:phoenix_pubsub, "~> 2.1"},

      # ============================================
      # HTTP & Networking
      # ============================================
      {:req, "~> 0.4"},
      {:finch, "~> 0.17"},
      {:mint, "~> 1.5"},
      {:castore, "~> 1.0"},

      # ============================================
      # Database & Storage
      # ============================================
      {:ecto_sql, "~> 3.11"},
      {:postgrex, ">= 0.0.0"},
      {:ecto_psql_extras, "~> 0.7"},

      # ============================================
      # Background Jobs & Pipelines
      # ============================================
      {:oban, "~> 2.16"},
      {:broadway, "~> 1.0"},
      {:broadway_rabbitmq, "~> 0.8"},
      {:broadway_kafka, "~> 0.4", optional: true},

      # ============================================
      # Distributed Systems
      # ============================================
      {:libcluster, "~> 3.3"},
      {:horde, "~> 0.8"},
      {:swarm, "~> 3.4"},
      {:partisan, "~> 5.0", optional: true},

      # ============================================
      # Machine Learning (Nx Ecosystem)
      # ============================================
      {:nx, "~> 0.6"},
      {:axon, "~> 0.6"},
      {:bumblebee, "~> 0.4"},
      {:exla, "~> 0.6", runtime: Mix.env() != :test},

      # ============================================
      # Serialization & Data
      # ============================================
      {:jason, "~> 1.4"},
      {:msgpax, "~> 2.4"},
      {:protobuf, "~> 0.12"},

      # ============================================
      # Telemetry & Monitoring
      # ============================================
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},
      {:opentelemetry, "~> 1.3"},
      {:opentelemetry_exporter, "~> 1.6"},
      {:opentelemetry_phoenix, "~> 1.1"},
      {:opentelemetry_ecto, "~> 1.2"},

      # ============================================
      # Security & Authentication
      # ============================================
      {:argon2_elixir, "~> 4.0"},
      {:guardian, "~> 2.3"},
      {:cors_plug, "~> 3.0"},

      # ============================================
      # Rate Limiting & Caching
      # ============================================
      {:hammer, "~> 6.1"},
      {:hammer_backend_redis, "~> 6.1"},
      {:nebulex, "~> 2.5"},
      {:nebulex_adapters_cachex, "~> 2.1"},
      {:nebulex_redis_adapter, "~> 2.3"},

      # ============================================
      # Utilities
      # ============================================
      {:timex, "~> 3.7"},
      {:elixir_uuid, "~> 1.2"},
      {:typed_struct, "~> 0.3"},
      {:nimble_options, "~> 1.0"},

      # ============================================
      # Development & Testing
      # ============================================
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:mox, "~> 1.1", only: :test},
      {:excoveralls, "~> 0.18", only: :test},
      {:stream_data, "~> 0.6", only: [:dev, :test]},
      {:benchee, "~> 1.2", only: :dev}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "ecto.setup", "assets.setup"],
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      "assets.setup": ["cmd npm install --prefix assets"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"],
      "test.watch": ["test.watch --stale"],
      lint: ["format --check-formatted", "credo --strict"],
      ci: ["lint", "test --cover", "dialyzer"]
    ]
  end

  defp releases do
    [
      faceless_video: [
        include_executables_for: [:unix],
        applications: [runtime_tools: :permanent],
        steps: [:assemble, :tar],
        strip_beams: [keep: ["Docs"]],
        cookie: "#{:crypto.strong_rand_bytes(32) |> Base.encode64()}"
      ]
    ]
  end

  defp docs do
    [
      main: "FacelessVideo",
      logo: "priv/static/images/logo.png",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "docs/architecture.md",
        "docs/deployment.md"
      ],
      groups_for_modules: [
        Core: [
          FacelessVideo,
          FacelessVideo.Application,
          FacelessVideo.Config
        ],
        Pipeline: [
          FacelessVideo.Pipeline.VideoProcessor,
          FacelessVideo.Pipeline.AudioProcessor,
          FacelessVideo.Pipeline.Supervisor
        ],
        Events: [
          FacelessVideo.Events.Broadcaster,
          FacelessVideo.Events.Handler
        ],
        Services: [
          FacelessVideo.Services.VideoCompositor,
          FacelessVideo.Services.FFmpegWorker
        ],
        AI: [
          FacelessVideo.AI.ImageGenerator,
          FacelessVideo.AI.AudioGenerator,
          FacelessVideo.AI.ScriptAnalyzer
        ]
      ],
      nest_modules_by_prefix: [
        FacelessVideo.Pipeline,
        FacelessVideo.Events,
        FacelessVideo.Services,
        FacelessVideo.AI
      ]
    ]
  end

  defp dialyzer do
    [
      plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
      plt_add_apps: [:mix, :ex_unit],
      flags: [
        :error_handling,
        :underspecs,
        :unknown
      ]
    ]
  end

  defp preferred_cli_env do
    [
      coveralls: :test,
      "coveralls.detail": :test,
      "coveralls.post": :test,
      "coveralls.html": :test
    ]
  end

  defp package do
    [
      name: "faceless_video",
      files: ~w(lib priv .formatter.exs mix.exs README* LICENSE* CHANGELOG*),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md"
      }
    ]
  end
end
