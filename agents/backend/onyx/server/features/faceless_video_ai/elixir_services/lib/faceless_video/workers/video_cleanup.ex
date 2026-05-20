defmodule FacelessVideo.Workers.VideoCleanup do
  @moduledoc """
  Background worker for cleaning up temporary video files and expired data.

  Scheduled via Oban to run periodically.
  """
  use Oban.Worker,
    queue: :maintenance,
    max_attempts: 3,
    unique: [period: 3600]

  require Logger

  @temp_dirs [
    "/tmp/faceless_video/output",
    "/tmp/faceless_video/images",
    "/tmp/faceless_video/audio",
    "/tmp/faceless_video/temp"
  ]

  @max_age_hours 24

  @impl Oban.Worker
  def perform(%Oban.Job{args: args}) do
    max_age = Map.get(args, "max_age_hours", @max_age_hours)
    dry_run = Map.get(args, "dry_run", false)

    Logger.info("[VideoCleanup] Starting cleanup, max_age: #{max_age}h, dry_run: #{dry_run}")

    results =
      @temp_dirs
      |> Enum.map(&cleanup_directory(&1, max_age, dry_run))
      |> Enum.reduce(%{files_deleted: 0, bytes_freed: 0, errors: []}, fn result, acc ->
        %{
          files_deleted: acc.files_deleted + result.files_deleted,
          bytes_freed: acc.bytes_freed + result.bytes_freed,
          errors: acc.errors ++ result.errors
        }
      end)

    Logger.info(
      "[VideoCleanup] Complete. Deleted: #{results.files_deleted} files, " <>
        "Freed: #{format_bytes(results.bytes_freed)}"
    )

    if Enum.empty?(results.errors) do
      :ok
    else
      {:error, "Cleanup completed with errors: #{inspect(results.errors)}"}
    end
  end

  @doc """
  Schedule a cleanup job.
  """
  def schedule(opts \\ []) do
    args = %{
      max_age_hours: Keyword.get(opts, :max_age_hours, @max_age_hours),
      dry_run: Keyword.get(opts, :dry_run, false)
    }

    %{args: args}
    |> new()
    |> Oban.insert()
  end

  defp cleanup_directory(dir, max_age_hours, dry_run) do
    if File.dir?(dir) do
      cutoff = DateTime.add(DateTime.utc_now(), -max_age_hours * 3600, :second)

      dir
      |> File.ls!()
      |> Enum.reduce(%{files_deleted: 0, bytes_freed: 0, errors: []}, fn file, acc ->
        path = Path.join(dir, file)
        cleanup_file(path, cutoff, dry_run, acc)
      end)
    else
      %{files_deleted: 0, bytes_freed: 0, errors: []}
    end
  rescue
    error ->
      Logger.error("[VideoCleanup] Error reading directory #{dir}: #{inspect(error)}")
      %{files_deleted: 0, bytes_freed: 0, errors: [{dir, error}]}
  end

  defp cleanup_file(path, cutoff, dry_run, acc) do
    case File.stat(path, time: :posix) do
      {:ok, %{mtime: mtime, size: size}} ->
        file_time = DateTime.from_unix!(mtime)

        if DateTime.compare(file_time, cutoff) == :lt do
          if dry_run do
            Logger.debug("[VideoCleanup] Would delete: #{path}")
          else
            case File.rm(path) do
              :ok ->
                Logger.debug("[VideoCleanup] Deleted: #{path}")

              {:error, reason} ->
                Logger.warning("[VideoCleanup] Failed to delete #{path}: #{reason}")
                acc = %{acc | errors: [{path, reason} | acc.errors]}
            end
          end

          %{acc | files_deleted: acc.files_deleted + 1, bytes_freed: acc.bytes_freed + size}
        else
          acc
        end

      {:error, reason} ->
        %{acc | errors: [{path, reason} | acc.errors]}
    end
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 2)} KB"
  defp format_bytes(bytes) when bytes < 1024 * 1024 * 1024, do: "#{Float.round(bytes / 1024 / 1024, 2)} MB"
  defp format_bytes(bytes), do: "#{Float.round(bytes / 1024 / 1024 / 1024, 2)} GB"
end




