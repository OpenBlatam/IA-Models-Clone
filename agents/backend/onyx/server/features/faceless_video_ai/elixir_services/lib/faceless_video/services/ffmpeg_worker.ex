defmodule FacelessVideo.Services.FFmpegWorker do
  @moduledoc """
  FFmpeg worker for video optimization and transcoding.

  Provides:
  - Video optimization for web delivery
  - Format conversion
  - Resolution scaling
  - Bitrate adjustment
  """

  require Logger

  @ffmpeg_path System.get_env("FFMPEG_PATH", "ffmpeg")

  @type optimize_opts :: [
          preset: String.t(),
          crf: integer(),
          max_bitrate: String.t(),
          audio_bitrate: String.t(),
          resolution: {integer(), integer()} | nil
        ]

  @doc """
  Optimize video for web delivery.

  Uses two-pass encoding for optimal quality/size ratio.
  """
  @spec optimize(String.t(), String.t(), map()) :: {:ok, String.t()} | {:error, term()}
  def optimize(input_path, output_path, options \\ %{}) do
    preset = options["preset"] || "medium"
    crf = options["crf"] || 23
    resolution = options["resolution"]

    args = build_optimize_args(input_path, output_path, preset, crf, resolution)

    Logger.info("[FFmpegWorker] Optimizing #{input_path}")

    case run_ffmpeg(args) do
      :ok ->
        Logger.info("[FFmpegWorker] Optimization complete: #{output_path}")
        {:ok, output_path}

      {:error, reason} = error ->
        Logger.error("[FFmpegWorker] Optimization failed: #{inspect(reason)}")
        error
    end
  end

  @doc """
  Transcode video to a different format.
  """
  @spec transcode(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def transcode(input_path, output_path, opts \\ []) do
    codec = Keyword.get(opts, :codec, "libx264")
    audio_codec = Keyword.get(opts, :audio_codec, "aac")

    args = [
      "-i", input_path,
      "-c:v", codec,
      "-c:a", audio_codec,
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok -> {:ok, output_path}
      error -> error
    end
  end

  @doc """
  Extract audio from video.
  """
  @spec extract_audio(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def extract_audio(video_path, output_path, opts \\ []) do
    format = Keyword.get(opts, :format, "mp3")
    bitrate = Keyword.get(opts, :bitrate, "192k")

    args = [
      "-i", video_path,
      "-vn",
      "-acodec", audio_codec_for_format(format),
      "-ab", bitrate,
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok -> {:ok, output_path}
      error -> error
    end
  end

  @doc """
  Generate video thumbnail.
  """
  @spec thumbnail(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def thumbnail(video_path, output_path, opts \\ []) do
    time = Keyword.get(opts, :time, "00:00:01")
    size = Keyword.get(opts, :size, "320x180")

    args = [
      "-i", video_path,
      "-ss", time,
      "-vframes", "1",
      "-s", size,
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok -> {:ok, output_path}
      error -> error
    end
  end

  @doc """
  Resize video to target resolution.
  """
  @spec resize(String.t(), String.t(), {integer(), integer()}) ::
          {:ok, String.t()} | {:error, term()}
  def resize(input_path, output_path, {width, height}) do
    args = [
      "-i", input_path,
      "-vf", "scale=#{width}:#{height}",
      "-c:a", "copy",
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok -> {:ok, output_path}
      error -> error
    end
  end

  # Private functions

  defp build_optimize_args(input, output, preset, crf, resolution) do
    base_args = [
      "-i", input,
      "-c:v", "libx264",
      "-preset", preset,
      "-crf", to_string(crf),
      "-c:a", "aac",
      "-b:a", "128k",
      "-movflags", "+faststart",
      "-pix_fmt", "yuv420p"
    ]

    resolution_args =
      case resolution do
        {w, h} -> ["-vf", "scale=#{w}:#{h}"]
        _ -> []
      end

    base_args ++ resolution_args ++ ["-y", output]
  end

  defp audio_codec_for_format("mp3"), do: "libmp3lame"
  defp audio_codec_for_format("aac"), do: "aac"
  defp audio_codec_for_format("opus"), do: "libopus"
  defp audio_codec_for_format("flac"), do: "flac"
  defp audio_codec_for_format(_), do: "aac"

  defp run_ffmpeg(args) do
    Logger.debug("[FFmpegWorker] Running: #{@ffmpeg_path} #{Enum.join(args, " ")}")

    case System.cmd(@ffmpeg_path, args, stderr_to_stdout: true) do
      {_, 0} -> :ok
      {output, code} -> {:error, %{code: code, output: output}}
    end
  end
end




