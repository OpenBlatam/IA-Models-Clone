defmodule FacelessVideo.AI.AudioGenerator do
  @moduledoc """
  AI-powered audio/speech generation service.

  Supports multiple TTS providers:
  - OpenAI TTS
  - ElevenLabs
  - Google TTS
  - Local models (via Bumblebee)
  """

  require Logger

  @behaviour FacelessVideo.AI.Provider

  @default_provider :openai
  @default_voice "alloy"

  @voice_map %{
    "male_1" => "alloy",
    "male_2" => "echo",
    "female_1" => "nova",
    "female_2" => "shimmer",
    "neutral" => "alloy"
  }

  @doc """
  Generate speech from text.

  ## Options

  - `:provider` - TTS provider (:openai, :elevenlabs, :google)
  - `:voice` - Voice ID or name
  - `:speed` - Speaking speed (0.25 to 4.0)
  - `:format` - Output format (mp3, wav, opus)
  """
  @spec generate_speech(String.t(), String.t(), map()) :: {:ok, String.t()} | {:error, term()}
  def generate_speech(text, voice \\ "neutral", options \\ %{}) do
    provider = Map.get(options, "provider", @default_provider)
    
    start_time = System.monotonic_time(:millisecond)
    Logger.info("[AudioGenerator] Generating speech with #{provider}")

    result = generate_with_provider(provider, text, voice, options)

    duration = System.monotonic_time(:millisecond) - start_time
    Logger.info("[AudioGenerator] Generation completed in #{duration}ms")

    result
  end

  @impl true
  def available? do
    openai_key = Application.get_env(:faceless_video, :openai)[:api_key]
    elevenlabs_key = Application.get_env(:faceless_video, :elevenlabs)[:api_key]
    
    openai_key != nil or elevenlabs_key != nil
  end

  # Private functions

  defp generate_with_provider(:openai, text, voice, options) do
    config = Application.get_env(:faceless_video, :openai, [])
    api_key = config[:api_key] || System.get_env("OPENAI_API_KEY")

    unless api_key do
      {:error, :missing_api_key}
    else
      generate_with_openai(text, voice, options, api_key)
    end
  end

  defp generate_with_provider(:elevenlabs, text, voice, options) do
    config = Application.get_env(:faceless_video, :elevenlabs, [])
    api_key = config[:api_key] || System.get_env("ELEVENLABS_API_KEY")

    unless api_key do
      {:error, :missing_api_key}
    else
      generate_with_elevenlabs(text, voice, options, api_key)
    end
  end

  defp generate_with_provider(:google, text, _voice, options) do
    generate_with_google(text, options)
  end

  defp generate_with_provider(provider, _text, _voice, _options) do
    {:error, {:unknown_provider, provider}}
  end

  defp generate_with_openai(text, voice, options, api_key) do
    openai_voice = Map.get(@voice_map, voice, @default_voice)
    speed = Map.get(options, "speed", 1.0) |> clamp(0.25, 4.0)

    body = %{
      model: "tts-1",
      input: text,
      voice: openai_voice,
      speed: speed
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    request =
      Finch.build(:post, "https://api.openai.com/v1/audio/speech",
        headers,
        Jason.encode!(body)
      )

    case Finch.request(request, FacelessVideo.Finch, receive_timeout: 120_000) do
      {:ok, %Finch.Response{status: 200, body: audio_data}} ->
        path = temp_audio_path("mp3")
        File.write!(path, audio_data)
        {:ok, path}

      {:ok, %Finch.Response{status: status, body: body}} ->
        Logger.error("[AudioGenerator] OpenAI error: #{status} - #{body}")
        {:error, {:api_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp generate_with_elevenlabs(text, voice_id, options, api_key) do
    # Default to a standard voice if not specified
    voice = voice_id || "21m00Tcm4TlvDq8ikWAM"
    speed = Map.get(options, "speed", 1.0)

    body = %{
      text: text,
      model_id: "eleven_monolingual_v1",
      voice_settings: %{
        stability: 0.5,
        similarity_boost: 0.75,
        speed: speed
      }
    }

    headers = [
      {"xi-api-key", api_key},
      {"Content-Type", "application/json"}
    ]

    url = "https://api.elevenlabs.io/v1/text-to-speech/#{voice}"

    request = Finch.build(:post, url, headers, Jason.encode!(body))

    case Finch.request(request, FacelessVideo.Finch, receive_timeout: 120_000) do
      {:ok, %Finch.Response{status: 200, body: audio_data}} ->
        path = temp_audio_path("mp3")
        File.write!(path, audio_data)
        {:ok, path}

      {:ok, %Finch.Response{status: status, body: body}} ->
        Logger.error("[AudioGenerator] ElevenLabs error: #{status} - #{body}")
        {:error, {:api_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp generate_with_google(text, _options) do
    # Use gTTS or Google Cloud TTS
    # This is a placeholder - in production would use Google Cloud TTS API
    {:error, :not_implemented}
  end

  defp temp_audio_path(ext) do
    dir = Application.get_env(:faceless_video, :temp_dir, "/tmp/faceless_video/audio")
    File.mkdir_p!(dir)
    Path.join(dir, "#{:erlang.unique_integer([:positive])}.#{ext}")
  end

  defp clamp(value, min, max) do
    value |> max(min) |> min(max)
  end
end




