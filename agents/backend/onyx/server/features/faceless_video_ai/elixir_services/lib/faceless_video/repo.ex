defmodule FacelessVideo.Repo do
  @moduledoc """
  Ecto repository for FacelessVideo.

  Provides database access with connection pooling and telemetry.
  """
  use Ecto.Repo,
    otp_app: :faceless_video,
    adapter: Ecto.Adapters.Postgres

  require Logger

  @doc """
  Dynamically loads the repository URL from the environment.
  """
  def init(_type, config) do
    {:ok, Keyword.put(config, :url, database_url(config))}
  end

  defp database_url(config) do
    config[:url] || System.get_env("DATABASE_URL")
  end

  @doc """
  Health check for database connectivity.
  """
  @spec health_check() :: :ok | {:error, term()}
  def health_check do
    case query("SELECT 1") do
      {:ok, _} -> :ok
      {:error, reason} -> {:error, reason}
    end
  rescue
    error -> {:error, error}
  end

  @doc """
  Execute query with automatic retry on connection errors.
  """
  def query_with_retry(sql, params \\ [], opts \\ [], retries \\ 3) do
    case query(sql, params, opts) do
      {:ok, result} ->
        {:ok, result}

      {:error, %DBConnection.ConnectionError{}} when retries > 0 ->
        Process.sleep(100 * (4 - retries))
        query_with_retry(sql, params, opts, retries - 1)

      error ->
        error
    end
  end
end




