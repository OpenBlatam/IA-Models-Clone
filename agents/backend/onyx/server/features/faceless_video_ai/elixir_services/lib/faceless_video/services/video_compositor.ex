defmodule FacelessVideo.Services.VideoCompositor do
  @moduledoc """
  Video composition service using FFmpeg.

  Handles:
  - Image-to-video conversion with Ken Burns effect
  - Video concatenation with transitions
  - Audio mixing
  - Subtitle embedding
  """

  require Logger

  @ffmpeg_path System.get_env("FFMPEG_PATH", "ffmpeg")
  @ffprobe_path System.get_env("FFPROBE_PATH", "ffprobe")
  @default_fps 30
  @default_crf 23

  @type composition_result :: {:ok, String.t()} | {:error, term()}

  @doc """
  Compose images and audio into a video.

  ## Options

  - `:fps` - Frames per second (default: 30)
  - `:resolution` - Output resolution as {width, height}
  - `:format` - Output format (default: "mp4")
  - `:quality` - CRF value 0-51 (default: 23)
  """
  @spec compose([map()], map(), String.t(), keyword()) :: composition_result()
  def compose(images, audio, output_path, opts \\ []) do
    fps = Keyword.get(opts, :fps, @default_fps)
    crf = Keyword.get(opts, :quality, @default_crf)

    with {:ok, image_clips} <- create_image_clips(images, fps, opts),
         {:ok, concat_video} <- concatenate_clips(image_clips),
         {:ok, with_audio} <- add_audio(concat_video, audio.path) do
      # Move to final location
      File.rename!(with_audio, output_path)
      {:ok, output_path}
    end
  rescue
    error ->
      Logger.error("[VideoCompositor] Composition failed: #{inspect(error)}")
      {:error, error}
  end

  @doc """
  Add subtitles to a video.
  """
  @spec add_subtitles(String.t(), [map()]) :: composition_result()
  def add_subtitles(video_path, subtitles) do
    srt_path = Path.rootname(video_path) <> ".srt"
    output_path = Path.rootname(video_path) <> "_sub.mp4"

    # Generate SRT file
    :ok = write_srt(subtitles, srt_path)

    # Embed subtitles
    args = [
      "-i", video_path,
      "-vf", "subtitles=#{srt_path}",
      "-c:a", "copy",
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok ->
        File.rm(srt_path)
        {:ok, output_path}

      error ->
        error
    end
  end

  @doc """
  Get video duration using FFprobe.
  """
  @spec get_duration(String.t()) :: {:ok, float()} | {:error, term()}
  def get_duration(video_path) do
    args = [
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "default=noprint_wrappers=1:nokey=1",
      video_path
    ]

    case System.cmd(@ffprobe_path, args, stderr_to_stdout: true) do
      {output, 0} ->
        duration = output |> String.trim() |> String.to_float()
        {:ok, duration}

      {error, _} ->
        {:error, error}
    end
  end

  @doc """
  Get video metadata.
  """
  @spec get_metadata(String.t()) :: {:ok, map()} | {:error, term()}
  def get_metadata(video_path) do
    args = [
      "-v", "quiet",
      "-print_format", "json",
      "-show_format",
      "-show_streams",
      video_path
    ]

    case System.cmd(@ffprobe_path, args, stderr_to_stdout: true) do
      {output, 0} ->
        {:ok, Jason.decode!(output)}

      {error, _} ->
        {:error, error}
    end
  end

  # Private functions

  defp create_image_clips(images, fps, opts) do
    clips =
      images
      |> Task.async_stream(
        fn %{path: path, segment_index: idx} = image ->
          duration = image[:duration] || 5.0
          output = temp_path("clip_#{idx}.mp4")
          
          case create_ken_burns_clip(path, output, duration, fps, opts) do
            :ok -> {:ok, %{index: idx, path: output}}
            error -> error
          end
        end,
        max_concurrency: 4,
        timeout: 60_000
      )
      |> Enum.reduce_while({:ok, []}, fn
        {:ok, {:ok, clip}}, {:ok, acc} -> {:cont, {:ok, [clip | acc]}}
        {:ok, {:error, _} = error}, _ -> {:halt, error}
        {:exit, reason}, _ -> {:halt, {:error, reason}}
      end)

    case clips do
      {:ok, list} -> {:ok, Enum.sort_by(list, & &1.index)}
      error -> error
    end
  end

  defp create_ken_burns_clip(image_path, output_path, duration, fps, _opts) do
    # Ken Burns effect with zoom and pan
    filter = """
    zoompan=z='min(zoom+0.001,1.2)':d=#{round(duration * fps)}:s=1920x1080:fps=#{fps},
    format=yuv420p
    """
    |> String.replace("\n", "")

    args = [
      "-loop", "1",
      "-i", image_path,
      "-vf", filter,
      "-t", to_string(duration),
      "-c:v", "libx264",
      "-preset", "fast",
      "-pix_fmt", "yuv420p",
      "-y",
      output_path
    ]

    run_ffmpeg(args)
  end

  defp concatenate_clips(clips) do
    list_file = temp_path("concat_list.txt")
    output_path = temp_path("concatenated.mp4")

    # Create file list for FFmpeg concat
    content =
      clips
      |> Enum.map(fn %{path: path} -> "file '#{path}'" end)
      |> Enum.join("\n")

    File.write!(list_file, content)

    args = [
      "-f", "concat",
      "-safe", "0",
      "-i", list_file,
      "-c", "copy",
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok ->
        # Cleanup temporary clips
        Enum.each(clips, fn %{path: path} -> File.rm(path) end)
        File.rm(list_file)
        {:ok, output_path}

      error ->
        error
    end
  end

  defp add_audio(video_path, audio_path) do
    output_path = temp_path("with_audio.mp4")

    args = [
      "-i", video_path,
      "-i", audio_path,
      "-c:v", "copy",
      "-c:a", "aac",
      "-map", "0:v:0",
      "-map", "1:a:0",
      "-shortest",
      "-y",
      output_path
    ]

    case run_ffmpeg(args) do
      :ok ->
        File.rm(video_path)
        {:ok, output_path}

      error ->
        error
    end
  end

  defp write_srt(subtitles, path) do
    content =
      subtitles
      |> Enum.with_index(1)
      |> Enum.map(fn {sub, idx} ->
        """
        #{idx}
        #{format_srt_time(sub.start_time)} --> #{format_srt_time(sub.end_time)}
        #{sub.text}
        """
      end)
      |> Enum.join("\n")

    File.write!(path, content)
  end

  defp format_srt_time(seconds) do
    hours = trunc(seconds / 3600)
    minutes = trunc(rem(trunc(seconds), 3600) / 60)
    secs = rem(trunc(seconds), 60)
    ms = trunc((seconds - trunc(seconds)) * 1000)

    :io_lib.format("~2..0B:~2..0B:~2..0B,~3..0B", [hours, minutes, secs, ms])
    |> to_string()
  end

  defp run_ffmpeg(args) do
    Logger.debug("[VideoCompositor] Running: #{@ffmpeg_path} #{Enum.join(args, " ")}")

    case System.cmd(@ffmpeg_path, args, stderr_to_stdout: true) do
      {_, 0} -> :ok
      {output, code} -> {:error, %{code: code, output: output}}
    end
  end

  defp temp_path(filename) do
    dir = Application.get_env(:faceless_video, :temp_dir, "/tmp/faceless_video")
    File.mkdir_p!(dir)
    Path.join(dir, "#{:erlang.unique_integer([:positive])}_#{filename}")
  end
end




