defmodule TranscriberAI.MixProject do
  @moduledoc """
  Mix project configuration for TranscriberAI - distributed transcription pipeline.
  """
  use Mix.Project

  @version "1.0.0"
  @source_url "https://github.com/blatam-academy/social-video-transcriber"

  def project do
    [
      app: :transcriber_ai,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      consolidate_protocols: Mix.env() != :dev,
      aliases: aliases(),
      deps: deps(),
      releases: releases(),
      name: "TranscriberAI",
      description: description(),
      source_url: @source_url,
      docs: docs(),
      dialyzer: dialyzer()
    ]
  end

  def application do
    [
      mod: {TranscriberAI.Application, []},
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
    Distributed transcription pipeline with fault-tolerance, real-time events,
    and AI-powered content analysis. Built on Broadway, Phoenix, and OTP.
    """
  end

  defp deps do
    [
      # Core Framework
      {:phoenix, "~> 1.7.10"},
      {:phoenix_html, "~> 4.0"},
      {:phoenix_live_view, "~> 0.20.1"},
      {:phoenix_live_dashboard, "~> 0.8.2"},
      {:phoenix_pubsub, "~> 2.1"},

      # HTTP & Networking
      {:req, "~> 0.4"},
      {:finch, "~> 0.17"},

      # Database & Storage
      {:ecto_sql, "~> 3.11"},
      {:postgrex, ">= 0.0.0"},

      # Background Jobs & Pipelines
      {:oban, "~> 2.16"},
      {:broadway, "~> 1.0"},

      # Distributed Systems
      {:libcluster, "~> 3.3"},
      {:horde, "~> 0.8"},

      # Machine Learning (Nx Ecosystem)
      {:nx, "~> 0.6"},
      {:bumblebee, "~> 0.4"},

      # Serialization
      {:jason, "~> 1.4"},
      {:msgpax, "~> 2.4"},

      # Telemetry
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},

      # Rate Limiting & Caching
      {:hammer, "~> 6.1"},
      {:nebulex, "~> 2.5"},

      # Utilities
      {:timex, "~> 3.7"},
      {:elixir_uuid, "~> 1.2"},

      # Development & Testing
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:mox, "~> 1.1", only: :test}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "ecto.setup"],
      "ecto.setup": ["ecto.create", "ecto.migrate"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"],
      lint: ["format --check-formatted", "credo --strict"],
      ci: ["lint", "test", "dialyzer"]
    ]
  end

  defp releases do
    [
      transcriber_ai: [
        include_executables_for: [:unix],
        applications: [runtime_tools: :permanent],
        steps: [:assemble, :tar]
      ]
    ]
  end

  defp docs do
    [
      main: "TranscriberAI",
      extras: ["README.md"]
    ]
  end

  defp dialyzer do
    [
      plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
      flags: [:error_handling, :unknown]
    ]
  end
end












