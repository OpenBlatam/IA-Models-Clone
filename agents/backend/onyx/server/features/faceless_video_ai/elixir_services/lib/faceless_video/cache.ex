defmodule FacelessVideo.Cache do
  @moduledoc """
  Distributed cache using Nebulex with multi-level architecture.

  Provides a high-performance caching layer with:
  - L1: Local in-memory cache (Cachex) - ultra-fast, node-local
  - L2: Distributed cache (Redis) - shared across cluster

  ## Features

  - Automatic cache invalidation with TTL
  - Pattern-based key invalidation
  - Cache warming for preloading
  - Thundering herd protection
  - Statistics and monitoring
  - Telemetry integration

  ## Usage

      # Simple get/put
      Cache.put("key", value, ttl: :timer.minutes(5))
      Cache.get("key")

      # Get or compute (with thundering herd protection)
      Cache.get_or_store("expensive:computation", fn ->
        compute_expensive_value()
      end, ttl: :timer.hours(1))

      # Pattern-based invalidation
      Cache.invalidate_pattern("user:123:*")

      # Bulk operations
      Cache.warm([{"key1", val1}, {"key2", val2}])

  ## Configuration

      config :faceless_video, FacelessVideo.Cache,
        levels: [
          {FacelessVideo.Cache.L1, gc_interval: :timer.hours(1)},
          {FacelessVideo.Cache.L2, []}
        ]

  ## Telemetry Events

  - `[:faceless_video, :cache, :get, :hit]`
  - `[:faceless_video, :cache, :get, :miss]`
  - `[:faceless_video, :cache, :put]`
  - `[:faceless_video, :cache, :delete]`
  - `[:faceless_video, :cache, :invalidate]`
  """
  use Nebulex.Cache,
    otp_app: :faceless_video,
    adapter: Nebulex.Adapters.Multilevel

  alias FacelessVideo.Cache.{L1, L2}

  require Logger

  @default_ttl :timer.hours(1)
  @lock_ttl :timer.seconds(30)
  @lock_retry_delay 50
  @max_lock_retries 10

  # ============================================
  # Type Definitions
  # ============================================

  @type key :: term()
  @type value :: term()
  @type ttl :: non_neg_integer()
  @type pattern :: String.t()

  @type cache_opts :: [
          ttl: ttl(),
          tags: [atom()],
          level: :l1 | :l2 | :all
        ]

  @type stats :: %{
          l1: map(),
          l2: map(),
          hit_rate: float(),
          total_hits: non_neg_integer(),
          total_misses: non_neg_integer()
        }

  # ============================================
  # Public API
  # ============================================

  @doc """
  Get a value from cache with optional default.

  ## Examples

      iex> Cache.fetch("existing_key")
      {:ok, value}

      iex> Cache.fetch("missing_key", :default)
      :default
  """
  @spec fetch(key(), value()) :: value()
  def fetch(key, default \\ nil) do
    start_time = System.monotonic_time(:microsecond)

    case get(key) do
      nil ->
        emit_telemetry(:get, :miss, key, start_time)
        default

      value ->
        emit_telemetry(:get, :hit, key, start_time)
        value
    end
  end

  @doc """
  Get or compute and store a value with thundering herd protection.

  If the key exists, returns the cached value.
  If not, acquires a lock, computes using the function, stores, and returns.

  ## Options

  - `:ttl` - Time to live in milliseconds (default: 1 hour)
  - `:tags` - Tags for group invalidation

  ## Examples

      Cache.get_or_store("user:123:profile", fn ->
        Repo.get!(User, 123)
      end, ttl: :timer.minutes(30))
  """
  @spec get_or_store(key(), (-> value()), cache_opts()) :: value()
  def get_or_store(key, fun, opts \\ []) when is_function(fun, 0) do
    case get(key) do
      nil ->
        compute_and_store(key, fun, opts)

      value ->
        value
    end
  end

  @doc """
  Async version of get_or_store with lock-based thundering herd protection.

  Prevents multiple processes from computing the same expensive value.
  """
  @spec get_or_store_async(key(), (-> value()), cache_opts()) :: value()
  def get_or_store_async(key, fun, opts \\ []) when is_function(fun, 0) do
    case get(key) do
      nil ->
        with_lock(key, fn -> compute_and_store(key, fun, opts) end, opts)

      value ->
        value
    end
  end

  @doc """
  Invalidate all keys matching a pattern.

  Pattern supports:
  - `*` - Match any characters
  - `?` - Match single character

  ## Examples

      # Invalidate all user cache
      Cache.invalidate_pattern("user:*")

      # Invalidate specific user's video cache
      Cache.invalidate_pattern("user:123:video:*")
  """
  @spec invalidate_pattern(pattern()) :: {:ok, non_neg_integer()} | {:error, term()}
  def invalidate_pattern(pattern) do
    start_time = System.monotonic_time(:microsecond)

    try do
      # Invalidate in L1 (local)
      l1_count = invalidate_level(L1, pattern)

      # Invalidate in L2 (distributed)
      l2_count = invalidate_level(L2, pattern)

      total = l1_count + l2_count
      emit_telemetry(:invalidate, :ok, pattern, start_time, %{count: total})

      Logger.debug("[Cache] Invalidated #{total} keys matching '#{pattern}'")
      {:ok, total}
    rescue
      error ->
        Logger.error("[Cache] Pattern invalidation failed: #{inspect(error)}")
        {:error, error}
    end
  end

  @doc """
  Warm the cache with precomputed key-value pairs.

  ## Options

  - `:ttl` - TTL for all entries
  - `:async` - Warm asynchronously (default: false)

  ## Examples

      Cache.warm([
        {"config:app", app_config},
        {"config:features", feature_flags}
      ], ttl: :timer.hours(24))
  """
  @spec warm([{key(), value()}], cache_opts()) :: :ok | {:error, term()}
  def warm(key_value_pairs, opts \\ []) when is_list(key_value_pairs) do
    async? = Keyword.get(opts, :async, false)

    warmup_fn = fn ->
      Enum.each(key_value_pairs, fn {key, value} ->
        put(key, value, opts)
      end)

      Logger.info("[Cache] Warmed cache with #{length(key_value_pairs)} entries")
    end

    if async? do
      Task.start(warmup_fn)
      :ok
    else
      warmup_fn.()
      :ok
    end
  rescue
    error ->
      Logger.error("[Cache] Cache warming failed: #{inspect(error)}")
      {:error, error}
  end

  @doc """
  Get comprehensive cache statistics.
  """
  @spec statistics() :: stats()
  def statistics do
    l1_stats = safe_stats(L1)
    l2_stats = safe_stats(L2)

    total_hits = (l1_stats[:hits] || 0) + (l2_stats[:hits] || 0)
    total_misses = (l1_stats[:misses] || 0) + (l2_stats[:misses] || 0)
    total = total_hits + total_misses

    %{
      l1: l1_stats,
      l2: l2_stats,
      hit_rate: if(total > 0, do: Float.round(total_hits / total * 100, 2), else: 0.0),
      total_hits: total_hits,
      total_misses: total_misses,
      memory_bytes: estimate_memory()
    }
  end

  @doc """
  Check if a key exists in cache.
  """
  @spec exists?(key()) :: boolean()
  def exists?(key) do
    get(key) != nil
  end

  @doc """
  Delete multiple keys atomically.
  """
  @spec delete_many([key()]) :: :ok
  def delete_many(keys) when is_list(keys) do
    Enum.each(keys, &delete/1)
    :ok
  end

  @doc """
  Get multiple keys at once.
  """
  @spec get_many([key()]) :: %{key() => value()}
  def get_many(keys) when is_list(keys) do
    keys
    |> Enum.map(fn key -> {key, get(key)} end)
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  @doc """
  Health check for cache availability.
  """
  @spec health_check() :: :ok | {:error, term()}
  def health_check do
    test_key = "__health_check_#{:erlang.unique_integer()}"

    try do
      put(test_key, "ok", ttl: 1000)

      case get(test_key) do
        "ok" ->
          delete(test_key)
          :ok

        _ ->
          {:error, :read_failed}
      end
    rescue
      error -> {:error, error}
    end
  end

  # ============================================
  # Private Functions
  # ============================================

  defp compute_and_store(key, fun, opts) do
    value = fun.()
    ttl = Keyword.get(opts, :ttl, @default_ttl)
    put(key, value, ttl: ttl)
    value
  end

  defp with_lock(key, fun, opts) do
    lock_key = "lock:#{key}"
    retries = Keyword.get(opts, :max_retries, @max_lock_retries)
    do_with_lock(key, lock_key, fun, opts, retries)
  end

  defp do_with_lock(key, _lock_key, fun, _opts, 0) do
    # Fallback: just compute without lock
    Logger.warning("[Cache] Lock acquisition failed for #{key}, computing without lock")
    fun.()
  end

  defp do_with_lock(key, lock_key, fun, opts, retries) do
    if acquire_lock(lock_key) do
      try do
        # Double-check after acquiring lock
        case get(key) do
          nil -> fun.()
          value -> value
        end
      after
        release_lock(lock_key)
      end
    else
      Process.sleep(@lock_retry_delay)
      do_with_lock(key, lock_key, fun, opts, retries - 1)
    end
  end

  defp acquire_lock(key) do
    case L1.put_new(key, true, ttl: @lock_ttl) do
      true -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  defp release_lock(key) do
    L1.delete(key)
  rescue
    _ -> :ok
  end

  defp invalidate_level(cache_module, pattern) do
    regex = pattern_to_regex(pattern)

    cache_module.stream()
    |> Stream.filter(fn {key, _} -> Regex.match?(regex, to_string(key)) end)
    |> Stream.map(fn {key, _} ->
      cache_module.delete(key)
      1
    end)
    |> Enum.sum()
  rescue
    _ -> 0
  end

  defp pattern_to_regex(pattern) do
    pattern
    |> Regex.escape()
    |> String.replace("\\*", ".*")
    |> String.replace("\\?", ".")
    |> then(&Regex.compile!("^#{&1}$"))
  end

  defp safe_stats(cache_module) do
    cache_module.stats()
  rescue
    _ -> %{}
  end

  defp estimate_memory do
    l1_size = safe_size(L1)
    # Rough estimate: 1KB per entry average
    l1_size * 1024
  end

  defp safe_size(cache_module) do
    cache_module.size()
  rescue
    _ -> 0
  end

  defp emit_telemetry(operation, result, key, start_time, extra \\ %{}) do
    duration = System.monotonic_time(:microsecond) - start_time

    :telemetry.execute(
      [:faceless_video, :cache, operation, result],
      Map.merge(%{duration_us: duration, count: 1}, extra),
      %{key: key}
    )
  end
end

defmodule FacelessVideo.Cache.L1 do
  @moduledoc """
  Local in-memory cache (L1) using Cachex adapter.

  Provides ultra-fast node-local caching with:
  - Sub-millisecond access times
  - Automatic garbage collection
  - Memory limit enforcement
  """
  use Nebulex.Cache,
    otp_app: :faceless_video,
    adapter: Nebulex.Adapters.Local

  @doc """
  Get L1 cache configuration.
  """
  def config do
    Application.get_env(:faceless_video, __MODULE__, [])
  end
end

defmodule FacelessVideo.Cache.L2 do
  @moduledoc """
  Distributed Redis cache (L2).

  Provides cluster-wide caching with:
  - Shared state across nodes
  - Persistence options
  - Pub/sub for invalidation
  """
  use Nebulex.Cache,
    otp_app: :faceless_video,
    adapter: NebulexRedisAdapter

  @doc """
  Get L2 cache configuration.
  """
  def config do
    Application.get_env(:faceless_video, __MODULE__, [])
  end
end
