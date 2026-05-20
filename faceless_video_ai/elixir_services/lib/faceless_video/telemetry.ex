defmodule FacelessVideoWeb.Telemetry do
  @moduledoc """
  Telemetry supervisor for FacelessVideo.

  Sets up telemetry handlers and metrics reporters.
  """
  use Supervisor
  import Telemetry.Metrics

  def start_link(arg) do
    Supervisor.start_link(__MODULE__, arg, name: __MODULE__)
  end

  @impl true
  def init(_arg) do
    children = [
      {:telemetry_poller, measurements: periodic_measurements(), period: 10_000}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end

  @doc """
  Returns all metrics to be tracked.
  """
  def metrics do
    [
      # Phoenix Metrics
      summary("phoenix.endpoint.start.system_time",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.endpoint.stop.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.start.system_time",
        tags: [:route],
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.stop.duration",
        tags: [:route],
        unit: {:native, :millisecond}
      ),

      # Database Metrics
      summary("faceless_video.repo.query.total_time",
        unit: {:native, :millisecond},
        description: "The sum of the other measurements"
      ),
      summary("faceless_video.repo.query.decode_time",
        unit: {:native, :millisecond},
        description: "Time spent decoding the data"
      ),
      summary("faceless_video.repo.query.query_time",
        unit: {:native, :millisecond},
        description: "Time spent executing the query"
      ),
      summary("faceless_video.repo.query.queue_time",
        unit: {:native, :millisecond},
        description: "Time spent waiting for a connection"
      ),

      # Broadway Pipeline Metrics
      counter("faceless_video.pipeline.message.start.count",
        tags: [:processor],
        description: "Number of messages started processing"
      ),
      summary("faceless_video.pipeline.message.stop.duration_us",
        tags: [:processor],
        unit: :microsecond,
        description: "Message processing duration"
      ),
      counter("faceless_video.pipeline.message.exception.count",
        tags: [:processor, :reason],
        description: "Number of message processing exceptions"
      ),
      counter("faceless_video.pipeline.batch.start.count",
        tags: [:batcher],
        description: "Number of batches started"
      ),
      summary("faceless_video.pipeline.batch.stop.duration_us",
        tags: [:batcher],
        unit: :microsecond,
        description: "Batch processing duration"
      ),

      # Event Broadcasting Metrics
      counter("faceless_video.events.broadcast.count",
        tags: [:topic, :event],
        description: "Number of events broadcast"
      ),
      summary("faceless_video.events.broadcast.duration_us",
        unit: :microsecond,
        description: "Event broadcast duration"
      ),
      counter("faceless_video.events.deliver.count",
        tags: [:topic],
        description: "Number of events delivered"
      ),

      # Cache Metrics
      counter("faceless_video.cache.get.hit.count",
        description: "Cache hits"
      ),
      counter("faceless_video.cache.get.miss.count",
        description: "Cache misses"
      ),
      counter("faceless_video.cache.put.count",
        description: "Cache puts"
      ),
      counter("faceless_video.cache.invalidate.count",
        description: "Cache invalidations"
      ),

      # Oban Job Metrics
      counter("oban.job.start.count",
        tags: [:queue, :worker],
        description: "Jobs started"
      ),
      summary("oban.job.stop.duration",
        tags: [:queue, :worker],
        unit: {:native, :millisecond},
        description: "Job execution duration"
      ),
      counter("oban.job.exception.count",
        tags: [:queue, :worker],
        description: "Job exceptions"
      ),

      # VM Metrics
      last_value("vm.memory.total", unit: :byte),
      last_value("vm.memory.processes", unit: :byte),
      last_value("vm.memory.system", unit: :byte),
      last_value("vm.total_run_queue_lengths.total"),
      last_value("vm.total_run_queue_lengths.cpu"),
      last_value("vm.total_run_queue_lengths.io"),
      last_value("vm.system_counts.process_count")
    ]
  end

  defp periodic_measurements do
    [
      {__MODULE__, :measure_memory, []},
      {__MODULE__, :measure_processes, []},
      {__MODULE__, :measure_run_queue, []}
    ]
  end

  @doc false
  def measure_memory do
    memory = :erlang.memory()

    :telemetry.execute([:vm, :memory], %{
      total: memory[:total],
      processes: memory[:processes],
      system: memory[:system],
      atom: memory[:atom],
      binary: memory[:binary],
      ets: memory[:ets]
    })
  end

  @doc false
  def measure_processes do
    :telemetry.execute([:vm, :system_counts], %{
      process_count: :erlang.system_info(:process_count),
      port_count: :erlang.system_info(:port_count)
    })
  end

  @doc false
  def measure_run_queue do
    total = :erlang.statistics(:total_run_queue_lengths_all)
    cpu = :erlang.statistics(:total_run_queue_lengths)
    io = total - cpu

    :telemetry.execute([:vm, :total_run_queue_lengths], %{
      total: total,
      cpu: cpu,
      io: io
    })
  end
end

defmodule FacelessVideo.Telemetry.Handlers do
  @moduledoc """
  Custom telemetry event handlers.
  """
  require Logger

  @doc """
  Attach all telemetry handlers.
  """
  def attach do
    :telemetry.attach_many(
      "faceless-video-handlers",
      [
        [:faceless_video, :pipeline, :message, :exception],
        [:faceless_video, :events, :broadcast],
        [:oban, :job, :exception]
      ],
      &handle_event/4,
      nil
    )
  end

  def handle_event([:faceless_video, :pipeline, :message, :exception], measurements, metadata, _) do
    Logger.error(
      "[Pipeline] Message processing exception: #{inspect(metadata.reason)}",
      processor: metadata[:processor]
    )
  end

  def handle_event([:faceless_video, :events, :broadcast], measurements, metadata, _) do
    Logger.debug(
      "[Events] Broadcast to #{metadata.topic}: #{metadata.event}",
      duration_us: measurements.duration_us
    )
  end

  def handle_event([:oban, :job, :exception], measurements, metadata, _) do
    Logger.error(
      "[Oban] Job exception in #{metadata.worker}: #{inspect(metadata.reason)}",
      queue: metadata.queue,
      attempt: metadata.attempt
    )
  end
end




