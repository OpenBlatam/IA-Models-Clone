defmodule FacelessVideo.AI.ImageGenerator do
  @moduledoc """
  AI-powered image generation service.

  Supports multiple providers:
  - OpenAI DALL-E
  - Stability AI
  - Local models (via Bumblebee)
  """

  require Logger

  @behaviour FacelessVideo.AI.Provider

  @default_provider :openai
  @default_size "1024x1024"

  @doc """
  Generate an image from a text prompt.

  ## Options

  - `:provider` - AI provider (:openai, :stability, :local)
  - `:size` - Image size ("1024x1024", "1792x1024", etc.)
  - `:style` - Image style (provider-specific)
  - `:quality` - Image quality ("standard", "hd")
  """
  @spec generate(String.t(), map()) :: {:ok, String.t()} | {:error, term()}
  def generate(prompt, options \\ %{}) do
    provider = Map.get(options, "provider", @default_provider)
    
    start_time = System.monotonic_time(:millisecond)
    Logger.info("[ImageGenerator] Generating image with #{provider}")

    result = generate_with_provider(provider, prompt, options)

    duration = System.monotonic_time(:millisecond) - start_time
    Logger.info("[ImageGenerator] Generation completed in #{duration}ms")

    result
  end

  @impl true
  def available? do
    openai_key = Application.get_env(:faceless_video, :openai)[:api_key]
    stability_key = Application.get_env(:faceless_video, :stability)[:api_key]
    
    openai_key != nil or stability_key != nil
  end

  # Private functions

  defp generate_with_provider(:openai, prompt, options) do
    config = Application.get_env(:faceless_video, :openai, [])
    api_key = config[:api_key] || System.get_env("OPENAI_API_KEY")

    unless api_key do
      {:error, :missing_api_key}
    else
      generate_with_openai(prompt, options, api_key)
    end
  end

  defp generate_with_provider(:stability, prompt, options) do
    config = Application.get_env(:faceless_video, :stability, [])
    api_key = config[:api_key] || System.get_env("STABILITY_API_KEY")

    unless api_key do
      {:error, :missing_api_key}
    else
      generate_with_stability(prompt, options, api_key)
    end
  end

  defp generate_with_provider(:local, prompt, options) do
    generate_with_local(prompt, options)
  end

  defp generate_with_provider(provider, _prompt, _options) do
    {:error, {:unknown_provider, provider}}
  end

  defp generate_with_openai(prompt, options, api_key) do
    size = Map.get(options, "size", @default_size)
    quality = Map.get(options, "quality", "standard")

    body = %{
      model: "dall-e-3",
      prompt: prompt,
      size: size,
      quality: quality,
      n: 1
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    request =
      Finch.build(:post, "https://api.openai.com/v1/images/generations",
        headers,
        Jason.encode!(body)
      )

    case Finch.request(request, FacelessVideo.Finch, receive_timeout: 120_000) do
      {:ok, %Finch.Response{status: 200, body: response_body}} ->
        response = Jason.decode!(response_body)
        image_url = get_in(response, ["data", Access.at(0), "url"])
        download_image(image_url)

      {:ok, %Finch.Response{status: status, body: body}} ->
        Logger.error("[ImageGenerator] OpenAI error: #{status} - #{body}")
        {:error, {:api_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp generate_with_stability(prompt, options, api_key) do
    size = parse_size(Map.get(options, "size", @default_size))

    body = %{
      text_prompts: [%{text: prompt}],
      cfg_scale: 7,
      height: size.height,
      width: size.width,
      samples: 1,
      steps: 30
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    request = Finch.build(:post, url, headers, Jason.encode!(body))

    case Finch.request(request, FacelessVideo.Finch, receive_timeout: 120_000) do
      {:ok, %Finch.Response{status: 200, body: response_body}} ->
        response = Jason.decode!(response_body)
        base64_image = get_in(response, ["artifacts", Access.at(0), "base64"])
        save_base64_image(base64_image)

      {:ok, %Finch.Response{status: status, body: body}} ->
        Logger.error("[ImageGenerator] Stability error: #{status} - #{body}")
        {:error, {:api_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp generate_with_local(_prompt, _options) do
    # Placeholder for local model generation
    # Would use Bumblebee with Stable Diffusion
    {:error, :not_implemented}
  end

  defp download_image(url) do
    request = Finch.build(:get, url)

    case Finch.request(request, FacelessVideo.Finch, receive_timeout: 60_000) do
      {:ok, %Finch.Response{status: 200, body: body}} ->
        path = temp_image_path()
        File.write!(path, body)
        {:ok, path}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp save_base64_image(base64) do
    path = temp_image_path()
    File.write!(path, Base.decode64!(base64))
    {:ok, path}
  end

  defp temp_image_path do
    dir = Application.get_env(:faceless_video, :temp_dir, "/tmp/faceless_video/images")
    File.mkdir_p!(dir)
    Path.join(dir, "#{:erlang.unique_integer([:positive])}.png")
  end

  defp parse_size(size) when is_binary(size) do
    [w, h] = String.split(size, "x") |> Enum.map(&String.to_integer/1)
    %{width: w, height: h}
  end

  defp parse_size(%{width: _, height: _} = size), do: size
end




