defmodule FacelessVideo.Workers.MetricsAggregator do
  @moduledoc """
  Background worker for aggregating and storing metrics.

  Collects metrics from various sources and persists aggregated data.
  """
  use Oban.Worker,
    queue: :maintenance,
    max_attempts: 3

  alias FacelessVideo.{Cache, Events.Broadcaster}

  require Logger

  @impl Oban.Worker
  def perform(%Oban.Job{args: args}) do
    period = Map.get(args, "period", "hourly")

    Logger.info("[MetricsAggregator] Starting #{period} aggregation")

    metrics = collect_metrics()
    aggregated = aggregate_metrics(metrics, period)

    # Store aggregated metrics
    :ok = store_metrics(aggregated, period)

    # Broadcast metrics update
    Broadcaster.broadcast("metrics", :updated, aggregated)

    Logger.info("[MetricsAggregator] Completed #{period} aggregation")
    :ok
  end

  @doc """
  Schedule a metrics aggregation job.
  """
  def schedule(period \\ "hourly") do
    %{period: period}
    |> new()
    |> Oban.insert()
  end

  defp collect_metrics do
    %{
      video_processing: collect_video_metrics(),
      api: collect_api_metrics(),
      cache: collect_cache_metrics(),
      system: collect_system_metrics(),
      timestamp: DateTime.utc_now()
    }
  end

  defp collect_video_metrics do
    # Collect from Oban
    %{
      jobs_completed: count_completed_jobs(:video_processing),
      jobs_failed: count_failed_jobs(:video_processing),
      avg_duration_ms: avg_job_duration(:video_processing)
    }
  end

  defp collect_api_metrics do
    %{
      requests_total: get_counter("http.requests.total"),
      requests_success: get_counter("http.requests.success"),
      requests_error: get_counter("http.requests.error"),
      avg_latency_ms: get_gauge("http.latency.avg")
    }
  end

  defp collect_cache_metrics do
    Cache.statistics()
  end

  defp collect_system_metrics do
    memory = :erlang.memory()

    %{
      memory_total: memory[:total],
      memory_processes: memory[:processes],
      memory_system: memory[:system],
      process_count: :erlang.system_info(:process_count),
      scheduler_utilization: get_scheduler_utilization()
    }
  end

  defp aggregate_metrics(metrics, period) do
    %{
      period: period,
      timestamp: metrics.timestamp,
      video: %{
        jobs_completed: metrics.video_processing.jobs_completed,
        jobs_failed: metrics.video_processing.jobs_failed,
        success_rate: calculate_success_rate(metrics.video_processing),
        avg_duration_ms: metrics.video_processing.avg_duration_ms
      },
      api: %{
        requests_total: metrics.api.requests_total,
        error_rate: calculate_error_rate(metrics.api),
        avg_latency_ms: metrics.api.avg_latency_ms
      },
      cache: %{
        hit_rate: metrics.cache.hit_rate,
        total_hits: metrics.cache.total_hits,
        total_misses: metrics.cache.total_misses
      },
      system: %{
        memory_mb: metrics.system.memory_total / 1024 / 1024,
        process_count: metrics.system.process_count,
        scheduler_utilization: metrics.system.scheduler_utilization
      }
    }
  end

  defp store_metrics(aggregated, period) do
    key = "metrics:#{period}:#{DateTime.to_unix(aggregated.timestamp)}"

    Cache.put(key, aggregated, ttl: ttl_for_period(period))
    :ok
  end

  defp ttl_for_period("hourly"), do: :timer.hours(168)  # 7 days
  defp ttl_for_period("daily"), do: :timer.hours(720)   # 30 days
  defp ttl_for_period(_), do: :timer.hours(24)

  defp calculate_success_rate(%{jobs_completed: completed, jobs_failed: failed}) do
    total = completed + failed
    if total > 0, do: Float.round(completed / total * 100, 2), else: 100.0
  end

  defp calculate_error_rate(%{requests_total: total, requests_error: errors}) do
    if total > 0, do: Float.round(errors / total * 100, 2), else: 0.0
  end

  # Placeholder implementations - would connect to actual metrics sources
  defp count_completed_jobs(_queue), do: 0
  defp count_failed_jobs(_queue), do: 0
  defp avg_job_duration(_queue), do: 0.0
  defp get_counter(_name), do: 0
  defp get_gauge(_name), do: 0.0

  defp get_scheduler_utilization do
    case :scheduler.utilization(1) do
      [{:total, util, _}] -> Float.round(util * 100, 2)
      _ -> 0.0
    end
  rescue
    _ -> 0.0
  end
end




