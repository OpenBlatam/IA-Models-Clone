defmodule TranscriberAI.Events.Broadcaster do
  @moduledoc """
  Event broadcaster for real-time notifications using Phoenix PubSub.
  
  Features:
  - Real-time job status updates
  - Webhook delivery with retries
  - Channel-based subscriptions
  - Event history and replay
  """
  use GenServer
  require Logger

  alias Phoenix.PubSub

  @pubsub TranscriberAI.PubSub
  @topic_prefix "transcriber:"
  @max_history 1000

  defstruct [:history, :subscribers, :webhook_urls]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def broadcast(event_type, payload) do
    GenServer.cast(__MODULE__, {:broadcast, event_type, payload})
  end

  def subscribe(topic) do
    PubSub.subscribe(@pubsub, topic_name(topic))
  end

  def unsubscribe(topic) do
    PubSub.unsubscribe(@pubsub, topic_name(topic))
  end

  def subscribe_to_job(job_id) do
    subscribe("job:#{job_id}")
  end

  def subscribe_to_all do
    subscribe("all")
  end

  def register_webhook(url, events \\ [:all]) do
    GenServer.call(__MODULE__, {:register_webhook, url, events})
  end

  def unregister_webhook(url) do
    GenServer.call(__MODULE__, {:unregister_webhook, url})
  end

  def get_history(limit \\ 100) do
    GenServer.call(__MODULE__, {:get_history, limit})
  end

  @impl true
  def init(_opts) do
    state = %__MODULE__{
      history: :queue.new(),
      subscribers: %{},
      webhook_urls: []
    }
    
    {:ok, state}
  end

  @impl true
  def handle_cast({:broadcast, event_type, payload}, state) do
    event = build_event(event_type, payload)
    
    PubSub.broadcast(@pubsub, topic_name("all"), event)
    
    if job_id = Map.get(payload, :job_id) do
      PubSub.broadcast(@pubsub, topic_name("job:#{job_id}"), event)
    end
    
    deliver_webhooks(event, state.webhook_urls)
    
    new_history = add_to_history(state.history, event)
    
    {:noreply, %{state | history: new_history}}
  end

  @impl true
  def handle_call({:register_webhook, url, events}, _from, state) do
    webhook = %{url: url, events: events, registered_at: DateTime.utc_now()}
    new_webhooks = [webhook | state.webhook_urls]
    
    Logger.info("Registered webhook: #{url} for events: #{inspect(events)}")
    
    {:reply, :ok, %{state | webhook_urls: new_webhooks}}
  end

  @impl true
  def handle_call({:unregister_webhook, url}, _from, state) do
    new_webhooks = Enum.reject(state.webhook_urls, fn w -> w.url == url end)
    
    Logger.info("Unregistered webhook: #{url}")
    
    {:reply, :ok, %{state | webhook_urls: new_webhooks}}
  end

  @impl true
  def handle_call({:get_history, limit}, _from, state) do
    history = state.history
    |> :queue.to_list()
    |> Enum.take(-limit)
    
    {:reply, history, state}
  end

  defp build_event(type, payload) do
    %{
      id: UUID.uuid4(),
      type: type,
      payload: payload,
      timestamp: DateTime.utc_now(),
      version: 1
    }
  end

  defp topic_name(topic) do
    "#{@topic_prefix}#{topic}"
  end

  defp add_to_history(queue, event) do
    queue = :queue.in(event, queue)
    
    if :queue.len(queue) > @max_history do
      {_, queue} = :queue.out(queue)
      queue
    else
      queue
    end
  end

  defp deliver_webhooks(event, webhooks) do
    webhooks
    |> Enum.filter(fn w -> :all in w.events or event.type in w.events end)
    |> Enum.each(fn webhook ->
      Task.Supervisor.start_child(TranscriberAI.TaskSupervisor, fn ->
        deliver_webhook(webhook.url, event)
      end)
    end)
  end

  defp deliver_webhook(url, event, retries \\ 3) do
    body = Jason.encode!(event)
    headers = [{"content-type", "application/json"}]
    
    case Req.post(url, body: body, headers: headers) do
      {:ok, %{status: status}} when status in 200..299 ->
        Logger.debug("Webhook delivered to #{url}")
        :ok
      
      {:ok, %{status: status}} ->
        Logger.warning("Webhook to #{url} returned #{status}")
        maybe_retry(url, event, retries)
      
      {:error, reason} ->
        Logger.error("Webhook to #{url} failed: #{inspect(reason)}")
        maybe_retry(url, event, retries)
    end
  end

  defp maybe_retry(_url, _event, 0), do: :failed
  defp maybe_retry(url, event, retries) do
    Process.sleep(1000 * (4 - retries))
    deliver_webhook(url, event, retries - 1)
  end
end












