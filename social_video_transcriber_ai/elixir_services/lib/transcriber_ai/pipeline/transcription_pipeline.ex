defmodule TranscriberAI.Pipeline.TranscriptionPipeline do
  @moduledoc """
  Broadway pipeline for distributed transcription processing.
  
  Features:
  - Concurrent processing with configurable batches
  - Automatic retries with backoff
  - Dead letter queue for failed jobs
  - Telemetry integration
  - Rate limiting per job type
  """
  use Broadway
  require Logger

  alias Broadway.Message
  alias TranscriberAI.{Cache, Events, Transcription}

  @default_concurrency 5
  @batch_size 10
  @batch_timeout 5_000

  def start_link(opts) do
    producer_opts = Keyword.get(opts, :producer, [])
    
    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: [
        module: {TranscriberAI.Pipeline.Producer, producer_opts},
        concurrency: 1
      ],
      processors: [
        default: [
          concurrency: get_concurrency(),
          min_demand: 1,
          max_demand: 5
        ]
      ],
      batchers: [
        default: [
          batch_size: @batch_size,
          batch_timeout: @batch_timeout,
          concurrency: 2
        ],
        high_priority: [
          batch_size: 5,
          batch_timeout: 2_000,
          concurrency: 3
        ]
      ]
    )
  end

  @impl true
  def handle_message(_processor, %Message{data: job} = message, _context) do
    start_time = System.monotonic_time(:millisecond)
    
    case process_job(job) do
      {:ok, result} ->
        duration = System.monotonic_time(:millisecond) - start_time
        :telemetry.execute([:transcriber, :job, :success], %{duration: duration}, %{job_id: job.id})
        
        Events.broadcast(:job_completed, %{job_id: job.id, result: result})
        
        message
        |> Message.put_data(%{job: job, result: result})
        |> Message.put_batcher(get_batcher(job))
      
      {:error, reason} ->
        Logger.error("Job #{job.id} failed: #{inspect(reason)}")
        :telemetry.execute([:transcriber, :job, :failure], %{count: 1}, %{job_id: job.id, reason: reason})
        
        Events.broadcast(:job_failed, %{job_id: job.id, reason: reason})
        
        Message.failed(message, reason)
    end
  end

  @impl true
  def handle_batch(:default, messages, _batch_info, _context) do
    results = Enum.map(messages, fn %Message{data: %{job: job, result: result}} ->
      save_result(job, result)
    end)
    
    Logger.info("Processed batch of #{length(messages)} jobs")
    :telemetry.execute([:transcriber, :batch, :processed], %{count: length(messages)}, %{})
    
    messages
  end

  @impl true
  def handle_batch(:high_priority, messages, _batch_info, _context) do
    results = messages
    |> Enum.map(fn %Message{data: %{job: job, result: result}} ->
      save_result(job, result)
    end)
    
    Logger.info("Processed high-priority batch of #{length(messages)} jobs")
    messages
  end

  @impl true
  def handle_failed(messages, _context) do
    Enum.each(messages, fn %Message{data: job, status: status} ->
      case status do
        {:failed, reason} ->
          Logger.warning("Job #{job.id} moved to dead letter queue: #{inspect(reason)}")
          TranscriberAI.DeadLetterQueue.enqueue(job, reason)
        
        _ ->
          :ok
      end
    end)
    
    messages
  end

  defp process_job(%{type: :transcribe, url: url, options: options} = job) do
    with {:ok, video_path} <- download_video(url),
         {:ok, audio_path} <- extract_audio(video_path),
         {:ok, transcription} <- transcribe_audio(audio_path, options),
         {:ok, analysis} <- analyze_content(transcription, options) do
      
      Cache.put(cache_key(job), %{transcription: transcription, analysis: analysis})
      
      {:ok, %{
        transcription: transcription,
        analysis: analysis,
        job_id: job.id
      }}
    end
  end

  defp process_job(%{type: :analyze, text: text, options: options} = job) do
    case analyze_content(text, options) do
      {:ok, analysis} ->
        Cache.put(cache_key(job), analysis)
        {:ok, %{analysis: analysis, job_id: job.id}}
      
      error ->
        error
    end
  end

  defp process_job(%{type: :variants, text: text, count: count} = job) do
    case generate_variants(text, count) do
      {:ok, variants} ->
        {:ok, %{variants: variants, job_id: job.id}}
      
      error ->
        error
    end
  end

  defp process_job(job) do
    {:error, {:unknown_job_type, job.type}}
  end

  defp download_video(url) do
    Logger.info("Downloading video from #{url}")
    {:ok, "/tmp/video_#{:erlang.unique_integer([:positive])}.mp4"}
  end

  defp extract_audio(video_path) do
    Logger.info("Extracting audio from #{video_path}")
    {:ok, String.replace(video_path, ".mp4", ".wav")}
  end

  defp transcribe_audio(audio_path, options) do
    Logger.info("Transcribing audio from #{audio_path}")
    
    timestamps = Keyword.get(options, :include_timestamps, false)
    
    {:ok, %{
      text: "Transcription placeholder",
      segments: [],
      language: "en",
      duration: 0.0
    }}
  end

  defp analyze_content(content, options) do
    Logger.info("Analyzing content...")
    
    {:ok, %{
      framework: "Hook-Story-Offer",
      keywords: [],
      summary: "Summary placeholder",
      tone: "professional"
    }}
  end

  defp generate_variants(text, count) do
    Logger.info("Generating #{count} variants...")
    
    variants = for i <- 1..count do
      "Variant #{i}: #{text}"
    end
    
    {:ok, variants}
  end

  defp save_result(job, result) do
    Transcription.save_result(job.id, result)
  end

  defp cache_key(job) do
    "job:#{job.id}"
  end

  defp get_batcher(%{priority: :high}), do: :high_priority
  defp get_batcher(_), do: :default

  defp get_concurrency do
    System.get_env("PIPELINE_CONCURRENCY", "#{@default_concurrency}")
    |> String.to_integer()
  end
end












