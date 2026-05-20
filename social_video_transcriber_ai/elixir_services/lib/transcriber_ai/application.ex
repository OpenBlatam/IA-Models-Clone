defmodule TranscriberAI.Application do
  @moduledoc """
  OTP Application for TranscriberAI - Distributed Transcription Pipeline.
  
  Features:
  - Distributed job processing with Broadway
  - Real-time events with Phoenix PubSub
  - Fault-tolerant supervisors with OTP
  - Cluster support with libcluster
  - Distributed cache with Nebulex
  """
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    Logger.info("Starting TranscriberAI Application...")

    children = [
      TranscriberAI.Repo,
      TranscriberAI.Cache,
      {Phoenix.PubSub, name: TranscriberAI.PubSub},
      TranscriberAI.Telemetry,
      TranscriberAI.RateLimiter,
      TranscriberAI.Pipeline.Supervisor,
      TranscriberAI.Events.Broadcaster,
      TranscriberAI.Queue.Supervisor,
      TranscriberAI.Cluster.Supervisor,
      TranscriberAIWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: TranscriberAI.Supervisor]
    
    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("TranscriberAI Application started successfully")
        {:ok, pid}
      
      {:error, reason} = error ->
        Logger.error("Failed to start TranscriberAI: #{inspect(reason)}")
        error
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("Stopping TranscriberAI Application...")
    :ok
  end

  @impl true
  def config_change(changed, _new, removed) do
    TranscriberAIWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end












