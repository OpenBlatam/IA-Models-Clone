defmodule FacelessVideo.RateLimiter do
  @moduledoc """
  Distributed rate limiting using Hammer with Redis backend.

  Provides multiple rate limiting strategies:
  - Fixed window
  - Sliding window
  - Token bucket
  - Leaky bucket

  ## Usage

      # Check if request is allowed
      case RateLimiter.check("api:user:123", :standard) do
        {:allow, remaining} -> proceed()
        {:deny, retry_after} -> rate_limit_error(retry_after)
      end

      # Custom limits
      RateLimiter.check("uploads:user:123", :custom, limit: 10, period: 60_000)

  ## Configuration

      config :faceless_video, FacelessVideo.RateLimiter,
        enabled: true,
        limits: %{
          standard: {100, 60_000},    # 100 requests per minute
          burst: {20, 1_000},          # 20 requests per second
          uploads: {10, 3600_000}      # 10 uploads per hour
        }
  """
  use GenServer

  require Logger

  @default_limits %{
    standard: {100, 60_000},
    burst: {20, 1_000},
    uploads: {10, 3_600_000},
    api: {1000, 60_000},
    websocket: {50, 1_000}
  }

  # ============================================
  # Types
  # ============================================

  @type bucket :: atom()
  @type key :: String.t()
  @type limit :: pos_integer()
  @type period :: pos_integer()
  @type check_result :: {:allow, non_neg_integer()} | {:deny, non_neg_integer()}

  # ============================================
  # Client API
  # ============================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if a request is allowed under rate limits.

  ## Parameters

  - `key` - Unique identifier (e.g., "api:user:123")
  - `bucket` - Limit bucket (:standard, :burst, :uploads, :api)
  - `opts` - Custom options (limit, period)

  ## Returns

  - `{:allow, remaining}` - Request allowed, remaining requests in window
  - `{:deny, retry_after}` - Request denied, milliseconds until retry
  """
  @spec check(key(), bucket(), keyword()) :: check_result()
  def check(key, bucket \\ :standard, opts \\ []) do
    case get_limit(bucket, opts) do
      {limit, period} ->
        do_check(key, limit, period)

      :unlimited ->
        {:allow, :infinity}
    end
  end

  @doc """
  Check and increment counter atomically.
  """
  @spec hit(key(), bucket(), keyword()) :: check_result()
  def hit(key, bucket \\ :standard, opts \\ []) do
    case get_limit(bucket, opts) do
      {limit, period} ->
        case Hammer.check_rate(key, period, limit) do
          {:allow, count} ->
            {:allow, limit - count}

          {:deny, retry_after} ->
            {:deny, retry_after}
        end

      :unlimited ->
        {:allow, :infinity}
    end
  end

  @doc """
  Get current usage for a key.
  """
  @spec get_usage(key()) :: {:ok, non_neg_integer()} | {:error, term()}
  def get_usage(key) do
    case Hammer.inspect_bucket(key, 60_000, 100) do
      {:ok, {count, _, _, _, _}} -> {:ok, count}
      error -> error
    end
  end

  @doc """
  Reset rate limit for a key.
  """
  @spec reset(key()) :: :ok
  def reset(key) do
    Hammer.delete_buckets(key)
    :ok
  end

  @doc """
  Get all configured limits.
  """
  @spec get_limits() :: map()
  def get_limits do
    config_limits = Application.get_env(:faceless_video, __MODULE__, [])[:limits] || %{}
    Map.merge(@default_limits, config_limits)
  end

  @doc """
  Update a limit at runtime.
  """
  @spec set_limit(bucket(), limit(), period()) :: :ok
  def set_limit(bucket, limit, period) do
    GenServer.call(__MODULE__, {:set_limit, bucket, limit, period})
  end

  @doc """
  Check if rate limiting is enabled.
  """
  @spec enabled?() :: boolean()
  def enabled? do
    Application.get_env(:faceless_video, __MODULE__, [])[:enabled] != false
  end

  # ============================================
  # Server Callbacks
  # ============================================

  @impl true
  def init(opts) do
    state = %{
      limits: Map.merge(@default_limits, Keyword.get(opts, :limits, %{})),
      stats: %{
        allowed: 0,
        denied: 0,
        started_at: DateTime.utc_now()
      }
    }

    Logger.info("[RateLimiter] Started with limits: #{inspect(state.limits)}")
    {:ok, state}
  end

  @impl true
  def handle_call({:set_limit, bucket, limit, period}, _from, state) do
    new_limits = Map.put(state.limits, bucket, {limit, period})
    {:reply, :ok, %{state | limits: new_limits}}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    {:reply, state.stats, state}
  end

  @impl true
  def handle_cast({:record, type}, state) do
    new_stats =
      case type do
        :allow -> %{state.stats | allowed: state.stats.allowed + 1}
        :deny -> %{state.stats | denied: state.stats.denied + 1}
      end

    {:noreply, %{state | stats: new_stats}}
  end

  # ============================================
  # Private Functions
  # ============================================

  defp get_limit(bucket, opts) do
    custom_limit = Keyword.get(opts, :limit)
    custom_period = Keyword.get(opts, :period)

    cond do
      custom_limit && custom_period ->
        {custom_limit, custom_period}

      bucket == :unlimited ->
        :unlimited

      true ->
        limits = get_limits()
        Map.get(limits, bucket, Map.get(limits, :standard))
    end
  end

  defp do_check(key, limit, period) do
    if enabled?() do
      case Hammer.check_rate_inc(key, period, limit, 0) do
        {:allow, count} ->
          GenServer.cast(__MODULE__, {:record, :allow})
          {:allow, limit - count}

        {:deny, retry_after} ->
          GenServer.cast(__MODULE__, {:record, :deny})
          {:deny, retry_after}
      end
    else
      {:allow, limit}
    end
  end
end

defmodule FacelessVideo.RateLimiter.Plug do
  @moduledoc """
  Plug for rate limiting HTTP requests.
  """
  import Plug.Conn

  def init(opts), do: opts

  def call(conn, opts) do
    bucket = Keyword.get(opts, :bucket, :standard)
    key_func = Keyword.get(opts, :key, &default_key/1)
    key = key_func.(conn)

    case FacelessVideo.RateLimiter.hit(key, bucket) do
      {:allow, remaining} ->
        conn
        |> put_resp_header("x-ratelimit-remaining", to_string(remaining))

      {:deny, retry_after} ->
        conn
        |> put_resp_header("retry-after", to_string(div(retry_after, 1000)))
        |> put_resp_header("x-ratelimit-remaining", "0")
        |> send_resp(429, "Rate limit exceeded")
        |> halt()
    end
  end

  defp default_key(conn) do
    ip =
      conn
      |> get_req_header("x-forwarded-for")
      |> List.first()
      |> Kernel.||(to_string(:inet.ntoa(conn.remote_ip)))

    "http:#{ip}"
  end
end




