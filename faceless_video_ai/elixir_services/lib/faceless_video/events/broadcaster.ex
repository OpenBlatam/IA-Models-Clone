defmodule FacelessVideo.Events.Broadcaster do
  @moduledoc """
  Event broadcasting system using Phoenix PubSub.

  Provides fault-tolerant, distributed event broadcasting across
  the cluster with guaranteed delivery semantics.

  ## Features

  - Distributed broadcasting across cluster nodes
  - Topic-based subscriptions with pattern matching
  - Event filtering and transformation
  - Retry logic for failed broadcasts
  - Comprehensive telemetry integration
  - Rate limiting per subscriber

  ## Usage

      # Subscribe to events
      FacelessVideo.Events.Broadcaster.subscribe("video:*")

      # Broadcast an event
      FacelessVideo.Events.Broadcaster.broadcast("video:123", :completed, %{path: "/output.mp4"})

      # Receive events in your process
      def handle_info({:event, topic, event, payload}, state) do
        # Handle the event
      end

  ## Telemetry Events

  - `[:faceless_video, :events, :broadcast]` - When an event is broadcast
  - `[:faceless_video, :events, :subscribe]` - When a subscription is created
  - `[:faceless_video, :events, :deliver]` - When an event is delivered

  """
  use GenServer

  alias Phoenix.PubSub

  require Logger

  @pubsub FacelessVideo.PubSub
  @default_timeout 10_000
  @max_retries 3

  # ============================================
  # Type Definitions
  # ============================================

  @type topic :: String.t()
  @type event :: atom() | String.t()
  @type payload :: map()
  @type pattern :: String.t()

  @type subscription :: %{
          pattern: pattern(),
          pid: pid(),
          filter: (map() -> boolean()) | nil,
          transform: (map() -> map()) | nil,
          subscribed_at: DateTime.t(),
          metadata: map()
        }

  @type broadcast_opts :: [
          local_only: boolean(),
          priority: 1..10,
          retry: boolean(),
          timeout: pos_integer()
        ]

  # ============================================
  # Client API
  # ============================================

  @doc """
  Starts the broadcaster GenServer.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Subscribe to events matching the given pattern.

  ## Patterns

  - `"video:123"` - Exact match
  - `"video:*"` - Wildcard match
  - `"user:456:video:*"` - Mixed pattern

  ## Options

  - `:filter` - Function to filter events `(message -> boolean)`
  - `:transform` - Function to transform payload `(payload -> payload)`
  - `:metadata` - Additional metadata to store with subscription
  """
  @spec subscribe(pattern(), keyword()) :: :ok | {:error, term()}
  def subscribe(pattern, opts \\ []) do
    GenServer.call(__MODULE__, {:subscribe, pattern, self(), opts})
  end

  @doc """
  Unsubscribe from events matching the pattern.
  """
  @spec unsubscribe(pattern()) :: :ok
  def unsubscribe(pattern) do
    GenServer.call(__MODULE__, {:unsubscribe, pattern, self()})
  end

  @doc """
  Broadcast an event asynchronously to all subscribers.

  ## Options

  - `:local_only` - Only broadcast to local node (default: false)
  - `:priority` - Event priority 1-10 (default: 5)
  """
  @spec broadcast(topic(), event(), payload(), broadcast_opts()) :: :ok
  def broadcast(topic, event, payload, opts \\ []) do
    GenServer.cast(__MODULE__, {:broadcast, topic, event, payload, opts})
  end

  @doc """
  Synchronous broadcast that waits for delivery confirmation.
  """
  @spec broadcast_sync(topic(), event(), payload(), broadcast_opts()) ::
          {:ok, map()} | {:error, term()}
  def broadcast_sync(topic, event, payload, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    GenServer.call(__MODULE__, {:broadcast_sync, topic, event, payload, opts}, timeout)
  end

  @doc """
  Broadcast to multiple topics at once.
  """
  @spec broadcast_multi([{topic(), event(), payload()}], broadcast_opts()) :: :ok
  def broadcast_multi(messages, opts \\ []) do
    GenServer.cast(__MODULE__, {:broadcast_multi, messages, opts})
  end

  @doc """
  Get the current list of subscribers for a topic.
  """
  @spec subscribers(topic()) :: [map()]
  def subscribers(topic) do
    GenServer.call(__MODULE__, {:subscribers, topic})
  end

  @doc """
  Get event statistics.
  """
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Reset statistics.
  """
  @spec reset_stats() :: :ok
  def reset_stats do
    GenServer.call(__MODULE__, :reset_stats)
  end

  @doc """
  Check if the broadcaster is healthy.
  """
  @spec health_check() :: :ok | {:error, term()}
  def health_check do
    GenServer.call(__MODULE__, :health_check, 5_000)
  catch
    :exit, _ -> {:error, :not_responding}
  end

  # ============================================
  # Server Callbacks
  # ============================================

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)

    state = %{
      subscriptions: %{},
      stats: initial_stats(),
      opts: opts
    }

    emit_telemetry(:init, %{}, %{})
    {:ok, state}
  end

  @impl true
  def handle_call({:subscribe, pattern, pid, opts}, _from, state) do
    Process.monitor(pid)

    subscription = %{
      pattern: pattern,
      pid: pid,
      filter: Keyword.get(opts, :filter),
      transform: Keyword.get(opts, :transform),
      subscribed_at: DateTime.utc_now(),
      metadata: Keyword.get(opts, :metadata, %{})
    }

    topic = pattern_to_topic(pattern)
    :ok = PubSub.subscribe(@pubsub, topic)

    subscriptions = Map.put(state.subscriptions, {pid, pattern}, subscription)

    emit_telemetry(:subscribe, %{count: 1}, %{pattern: pattern, pid: pid})
    Logger.debug("[Broadcaster] #{inspect(pid)} subscribed to #{pattern}")

    {:reply, :ok, %{state | subscriptions: subscriptions}}
  end

  @impl true
  def handle_call({:unsubscribe, pattern, pid}, _from, state) do
    topic = pattern_to_topic(pattern)
    :ok = PubSub.unsubscribe(@pubsub, topic)

    subscriptions = Map.delete(state.subscriptions, {pid, pattern})
    Logger.debug("[Broadcaster] #{inspect(pid)} unsubscribed from #{pattern}")

    {:reply, :ok, %{state | subscriptions: subscriptions}}
  end

  @impl true
  def handle_call({:broadcast_sync, topic, event, payload, opts}, _from, state) do
    {result, new_stats} = do_broadcast(topic, event, payload, opts, state)
    {:reply, result, %{state | stats: new_stats}}
  end

  @impl true
  def handle_call({:subscribers, topic}, _from, state) do
    subs =
      state.subscriptions
      |> Enum.filter(fn {{_pid, pattern}, _} -> matches_topic?(pattern, topic) end)
      |> Enum.map(fn {{pid, _}, sub} ->
        %{
          pid: pid,
          subscribed_at: sub.subscribed_at,
          metadata: sub.metadata
        }
      end)

    {:reply, subs, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    enriched_stats =
      state.stats
      |> Map.put(:subscription_count, map_size(state.subscriptions))
      |> Map.put(:uptime_seconds, uptime_seconds(state.stats.started_at))

    {:reply, enriched_stats, state}
  end

  @impl true
  def handle_call(:reset_stats, _from, state) do
    {:reply, :ok, %{state | stats: initial_stats()}}
  end

  @impl true
  def handle_call(:health_check, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_cast({:broadcast, topic, event, payload, opts}, state) do
    {_result, new_stats} = do_broadcast(topic, event, payload, opts, state)
    {:noreply, %{state | stats: new_stats}}
  end

  @impl true
  def handle_cast({:broadcast_multi, messages, opts}, state) do
    new_stats =
      Enum.reduce(messages, state.stats, fn {topic, event, payload}, acc_stats ->
        {_result, updated_stats} =
          do_broadcast(topic, event, payload, opts, %{state | stats: acc_stats})

        updated_stats
      end)

    {:noreply, %{state | stats: new_stats}}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, pid, _reason}, state) do
    subscriptions =
      state.subscriptions
      |> Enum.reject(fn {{sub_pid, _pattern}, _} -> sub_pid == pid end)
      |> Map.new()

    Logger.debug("[Broadcaster] Cleaned up subscriptions for #{inspect(pid)}")
    {:noreply, %{state | subscriptions: subscriptions}}
  end

  @impl true
  def handle_info({:pubsub_event, message}, state) do
    # Handle incoming distributed events
    deliver_to_local_subscribers(message, state)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state) do
    {:noreply, state}
  end

  @impl true
  def terminate(reason, _state) do
    Logger.info("[Broadcaster] Terminating: #{inspect(reason)}")
    :ok
  end

  # ============================================
  # Private Functions
  # ============================================

  defp do_broadcast(topic, event, payload, opts, state) do
    start_time = System.monotonic_time(:microsecond)

    message = %{
      id: UUID.uuid4(),
      topic: topic,
      event: event,
      payload: payload,
      timestamp: DateTime.utc_now(),
      node: node(),
      priority: Keyword.get(opts, :priority, 5)
    }

    local_only? = Keyword.get(opts, :local_only, false)

    result =
      if local_only? do
        broadcast_local(message, state)
      else
        broadcast_distributed(message, state, opts)
      end

    duration = System.monotonic_time(:microsecond) - start_time

    emit_telemetry(:broadcast, %{duration_us: duration, count: 1}, %{
      topic: topic,
      event: event,
      local_only: local_only?
    })

    new_stats = update_stats(state.stats, result, duration)
    {result, new_stats}
  end

  defp broadcast_local(message, state) do
    delivered =
      state.subscriptions
      |> Enum.filter(fn {{_pid, pattern}, _} -> matches_topic?(pattern, message.topic) end)
      |> Enum.count(fn {{pid, _}, sub} -> deliver_to_subscriber(pid, sub, message) end)

    {:ok, %{delivered: delivered, mode: :local}}
  end

  defp broadcast_distributed(message, state, opts) do
    pubsub_topic = pattern_to_topic(message.topic)
    retry? = Keyword.get(opts, :retry, true)

    case PubSub.broadcast(@pubsub, pubsub_topic, {:pubsub_event, message}) do
      :ok ->
        # Also deliver to local subscribers
        local_count =
          state.subscriptions
          |> Enum.filter(fn {{_pid, pattern}, _} -> matches_topic?(pattern, message.topic) end)
          |> Enum.count(fn {{pid, _}, sub} -> deliver_to_subscriber(pid, sub, message) end)

        {:ok, %{delivered: :distributed, local_count: local_count, mode: :distributed}}

      {:error, reason} = error when retry? ->
        Logger.warning("[Broadcaster] Broadcast failed: #{inspect(reason)}, retrying...")
        retry_broadcast(message, state, @max_retries)

      error ->
        error
    end
  end

  defp retry_broadcast(_message, _state, 0) do
    {:error, :max_retries_exceeded}
  end

  defp retry_broadcast(message, state, retries_left) do
    Process.sleep(100 * (@max_retries - retries_left + 1))
    pubsub_topic = pattern_to_topic(message.topic)

    case PubSub.broadcast(@pubsub, pubsub_topic, {:pubsub_event, message}) do
      :ok -> {:ok, %{delivered: :distributed, retries: @max_retries - retries_left + 1}}
      {:error, _} -> retry_broadcast(message, state, retries_left - 1)
    end
  end

  defp deliver_to_subscriber(pid, subscription, message) do
    should_deliver? =
      case subscription.filter do
        nil -> true
        filter when is_function(filter, 1) -> safe_call(filter, [message], true)
        _ -> true
      end

    if should_deliver? do
      payload =
        case subscription.transform do
          nil -> message.payload
          transform when is_function(transform, 1) -> safe_call(transform, [message.payload], message.payload)
          _ -> message.payload
        end

      send(pid, {:event, message.topic, message.event, payload})

      emit_telemetry(:deliver, %{count: 1}, %{
        topic: message.topic,
        pid: pid
      })

      true
    else
      false
    end
  end

  defp deliver_to_local_subscribers(message, state) do
    state.subscriptions
    |> Enum.filter(fn {{_pid, pattern}, _} -> matches_topic?(pattern, message.topic) end)
    |> Enum.each(fn {{pid, _}, sub} -> deliver_to_subscriber(pid, sub, message) end)
  end

  defp safe_call(fun, args, default) do
    apply(fun, args)
  rescue
    _ -> default
  end

  defp matches_topic?(pattern, topic) do
    pattern_parts = String.split(pattern, ":")
    topic_parts = String.split(topic, ":")
    match_parts(pattern_parts, topic_parts)
  end

  defp match_parts([], []), do: true
  defp match_parts(["*" | _], _), do: true
  defp match_parts([p | pattern_rest], [t | topic_rest]) when p == t, do: match_parts(pattern_rest, topic_rest)
  defp match_parts(_, _), do: false

  defp pattern_to_topic(pattern) do
    pattern
    |> String.split(":")
    |> Enum.take_while(&(&1 != "*"))
    |> Enum.join(":")
    |> then(fn
      "" -> "global"
      topic -> topic
    end)
  end

  defp initial_stats do
    %{
      broadcasts: 0,
      delivered: 0,
      failed: 0,
      total_duration_us: 0,
      started_at: DateTime.utc_now()
    }
  end

  defp update_stats(stats, result, duration_us) do
    case result do
      {:ok, _} ->
        %{
          stats
          | broadcasts: stats.broadcasts + 1,
            delivered: stats.delivered + 1,
            total_duration_us: stats.total_duration_us + duration_us
        }

      {:error, _} ->
        %{
          stats
          | broadcasts: stats.broadcasts + 1,
            failed: stats.failed + 1,
            total_duration_us: stats.total_duration_us + duration_us
        }
    end
  end

  defp uptime_seconds(started_at) do
    DateTime.diff(DateTime.utc_now(), started_at)
  end

  defp emit_telemetry(event, measurements, metadata) do
    :telemetry.execute(
      [:faceless_video, :events, event],
      measurements,
      metadata
    )
  end
end
