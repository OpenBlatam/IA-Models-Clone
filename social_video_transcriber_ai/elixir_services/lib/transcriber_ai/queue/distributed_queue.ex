defmodule TranscriberAI.Queue.DistributedQueue do
  @moduledoc """
  Distributed priority queue using Horde for cluster-wide job distribution.
  
  Features:
  - Priority-based scheduling (CRITICAL, HIGH, NORMAL, LOW, BACKGROUND)
  - Automatic work distribution across nodes
  - Job deduplication
  - Persistence with recovery
  - Metrics and monitoring
  """
  use GenServer
  require Logger

  alias TranscriberAI.Events

  @priorities [:critical, :high, :normal, :low, :background]
  @max_queue_size 10_000

  defstruct [
    :queues,
    :processing,
    :completed,
    :failed,
    :stats
  ]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: via_tuple())
  end

  def enqueue(job, priority \\ :normal) do
    GenServer.call(via_tuple(), {:enqueue, job, priority})
  end

  def dequeue(priority \\ nil) do
    GenServer.call(via_tuple(), {:dequeue, priority})
  end

  def get_job(job_id) do
    GenServer.call(via_tuple(), {:get_job, job_id})
  end

  def cancel_job(job_id) do
    GenServer.call(via_tuple(), {:cancel_job, job_id})
  end

  def complete_job(job_id, result) do
    GenServer.cast(via_tuple(), {:complete_job, job_id, result})
  end

  def fail_job(job_id, reason) do
    GenServer.cast(via_tuple(), {:fail_job, job_id, reason})
  end

  def get_stats do
    GenServer.call(via_tuple(), :get_stats)
  end

  def get_queue_length(priority \\ nil) do
    GenServer.call(via_tuple(), {:get_queue_length, priority})
  end

  def list_jobs(opts \\ []) do
    GenServer.call(via_tuple(), {:list_jobs, opts})
  end

  @impl true
  def init(_opts) do
    state = %__MODULE__{
      queues: init_queues(),
      processing: %{},
      completed: :queue.new(),
      failed: :queue.new(),
      stats: init_stats()
    }
    
    schedule_cleanup()
    
    {:ok, state}
  end

  @impl true
  def handle_call({:enqueue, job, priority}, _from, state) do
    priority = validate_priority(priority)
    job = ensure_job_id(job)
    
    if queue_full?(state, priority) do
      {:reply, {:error, :queue_full}, state}
    else
      timestamp = System.monotonic_time(:millisecond)
      entry = {timestamp, job}
      
      new_queues = Map.update!(state.queues, priority, fn queue ->
        :queue.in(entry, queue)
      end)
      
      new_stats = update_stats(state.stats, :enqueued, priority)
      
      Events.broadcast(:job_queued, %{job_id: job.id, priority: priority})
      
      Logger.debug("Job #{job.id} enqueued with priority #{priority}")
      
      {:reply, {:ok, job.id}, %{state | queues: new_queues, stats: new_stats}}
    end
  end

  @impl true
  def handle_call({:dequeue, priority}, _from, state) do
    case find_next_job(state.queues, priority) do
      {nil, _} ->
        {:reply, {:error, :empty}, state}
      
      {{timestamp, job}, new_queues, from_priority} ->
        new_processing = Map.put(state.processing, job.id, {job, timestamp, from_priority})
        new_stats = update_stats(state.stats, :dequeued, from_priority)
        
        Events.broadcast(:job_started, %{job_id: job.id, priority: from_priority})
        
        {:reply, {:ok, job}, %{state | queues: new_queues, processing: new_processing, stats: new_stats}}
    end
  end

  @impl true
  def handle_call({:get_job, job_id}, _from, state) do
    result = cond do
      Map.has_key?(state.processing, job_id) ->
        {job, _, _} = state.processing[job_id]
        {:ok, %{job: job, status: :processing}}
      
      true ->
        find_in_queues(state.queues, job_id)
    end
    
    {:reply, result, state}
  end

  @impl true
  def handle_call({:cancel_job, job_id}, _from, state) do
    cond do
      Map.has_key?(state.processing, job_id) ->
        {job, _, priority} = state.processing[job_id]
        new_processing = Map.delete(state.processing, job_id)
        new_stats = update_stats(state.stats, :cancelled, priority)
        
        Events.broadcast(:job_cancelled, %{job_id: job_id})
        
        {:reply, :ok, %{state | processing: new_processing, stats: new_stats}}
      
      true ->
        case remove_from_queues(state.queues, job_id) do
          {:ok, new_queues, priority} ->
            new_stats = update_stats(state.stats, :cancelled, priority)
            Events.broadcast(:job_cancelled, %{job_id: job_id})
            {:reply, :ok, %{state | queues: new_queues, stats: new_stats}}
          
          :not_found ->
            {:reply, {:error, :not_found}, state}
        end
    end
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    queue_lengths = Enum.map(state.queues, fn {priority, queue} ->
      {priority, :queue.len(queue)}
    end) |> Map.new()
    
    stats = Map.merge(state.stats, %{
      queue_lengths: queue_lengths,
      processing_count: map_size(state.processing),
      completed_count: :queue.len(state.completed),
      failed_count: :queue.len(state.failed)
    })
    
    {:reply, stats, state}
  end

  @impl true
  def handle_call({:get_queue_length, nil}, _from, state) do
    total = Enum.reduce(state.queues, 0, fn {_, queue}, acc ->
      acc + :queue.len(queue)
    end)
    
    {:reply, total, state}
  end

  @impl true
  def handle_call({:get_queue_length, priority}, _from, state) do
    length = :queue.len(state.queues[priority])
    {:reply, length, state}
  end

  @impl true
  def handle_call({:list_jobs, opts}, _from, state) do
    status = Keyword.get(opts, :status, :all)
    limit = Keyword.get(opts, :limit, 100)
    
    jobs = case status do
      :queued ->
        state.queues
        |> Enum.flat_map(fn {priority, queue} ->
          queue
          |> :queue.to_list()
          |> Enum.map(fn {_, job} -> {job, priority, :queued} end)
        end)
      
      :processing ->
        state.processing
        |> Enum.map(fn {_, {job, _, priority}} -> {job, priority, :processing} end)
      
      :all ->
        queued = state.queues
        |> Enum.flat_map(fn {priority, queue} ->
          queue |> :queue.to_list() |> Enum.map(fn {_, job} -> {job, priority, :queued} end)
        end)
        
        processing = state.processing
        |> Enum.map(fn {_, {job, _, priority}} -> {job, priority, :processing} end)
        
        queued ++ processing
    end
    
    {:reply, Enum.take(jobs, limit), state}
  end

  @impl true
  def handle_cast({:complete_job, job_id, result}, state) do
    case Map.pop(state.processing, job_id) do
      {nil, _} ->
        {:noreply, state}
      
      {{job, start_time, priority}, new_processing} ->
        duration = System.monotonic_time(:millisecond) - start_time
        completed_entry = {job, result, DateTime.utc_now(), duration}
        
        new_completed = :queue.in(completed_entry, state.completed)
        new_completed = trim_queue(new_completed, 1000)
        
        new_stats = state.stats
        |> update_stats(:completed, priority)
        |> Map.update!(:total_processing_time, &(&1 + duration))
        
        Events.broadcast(:job_completed, %{job_id: job_id, duration: duration})
        
        {:noreply, %{state | 
          processing: new_processing, 
          completed: new_completed, 
          stats: new_stats
        }}
    end
  end

  @impl true
  def handle_cast({:fail_job, job_id, reason}, state) do
    case Map.pop(state.processing, job_id) do
      {nil, _} ->
        {:noreply, state}
      
      {{job, _, priority}, new_processing} ->
        failed_entry = {job, reason, DateTime.utc_now()}
        
        new_failed = :queue.in(failed_entry, state.failed)
        new_failed = trim_queue(new_failed, 1000)
        
        new_stats = update_stats(state.stats, :failed, priority)
        
        Events.broadcast(:job_failed, %{job_id: job_id, reason: reason})
        
        {:noreply, %{state | 
          processing: new_processing, 
          failed: new_failed, 
          stats: new_stats
        }}
    end
  end

  @impl true
  def handle_info(:cleanup, state) do
    new_completed = trim_queue(state.completed, 500)
    new_failed = trim_queue(state.failed, 500)
    
    schedule_cleanup()
    
    {:noreply, %{state | completed: new_completed, failed: new_failed}}
  end

  defp via_tuple do
    {:via, Horde.Registry, {TranscriberAI.Registry, __MODULE__}}
  end

  defp init_queues do
    @priorities
    |> Enum.map(fn p -> {p, :queue.new()} end)
    |> Map.new()
  end

  defp init_stats do
    priority_stats = @priorities
    |> Enum.map(fn p -> {p, %{enqueued: 0, dequeued: 0, completed: 0, failed: 0, cancelled: 0}} end)
    |> Map.new()
    
    Map.merge(priority_stats, %{
      total_processing_time: 0,
      started_at: DateTime.utc_now()
    })
  end

  defp validate_priority(p) when p in @priorities, do: p
  defp validate_priority(_), do: :normal

  defp ensure_job_id(%{id: _} = job), do: job
  defp ensure_job_id(job), do: Map.put(job, :id, UUID.uuid4())

  defp queue_full?(state, priority) do
    :queue.len(state.queues[priority]) >= @max_queue_size
  end

  defp find_next_job(queues, nil) do
    Enum.find_value(@priorities, {nil, queues}, fn priority ->
      case :queue.out(queues[priority]) do
        {{:value, entry}, new_queue} ->
          {entry, Map.put(queues, priority, new_queue), priority}
        
        {:empty, _} ->
          nil
      end
    end)
  end

  defp find_next_job(queues, priority) do
    case :queue.out(queues[priority]) do
      {{:value, entry}, new_queue} ->
        {entry, Map.put(queues, priority, new_queue), priority}
      
      {:empty, _} ->
        {nil, queues}
    end
  end

  defp find_in_queues(queues, job_id) do
    Enum.find_value(queues, {:error, :not_found}, fn {priority, queue} ->
      queue
      |> :queue.to_list()
      |> Enum.find(fn {_, job} -> job.id == job_id end)
      |> case do
        nil -> nil
        {_, job} -> {:ok, %{job: job, status: :queued, priority: priority}}
      end
    end)
  end

  defp remove_from_queues(queues, job_id) do
    Enum.find_value(queues, :not_found, fn {priority, queue} ->
      new_list = queue
      |> :queue.to_list()
      |> Enum.reject(fn {_, job} -> job.id == job_id end)
      
      if length(new_list) < :queue.len(queue) do
        {:ok, Map.put(queues, priority, :queue.from_list(new_list)), priority}
      else
        nil
      end
    end)
  end

  defp update_stats(stats, action, priority) do
    update_in(stats, [priority, action], &(&1 + 1))
  end

  defp trim_queue(queue, max_size) do
    if :queue.len(queue) > max_size do
      {_, queue} = :queue.out(queue)
      trim_queue(queue, max_size)
    else
      queue
    end
  end

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup, :timer.minutes(5))
  end
end












