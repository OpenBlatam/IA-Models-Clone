# Elixir Services - Distributed Real-Time Pipeline

⚡ Servicios distribuidos en Elixir para **Social Video Transcriber AI**.

## Características

### 🔄 Broadway Pipeline
- Procesamiento concurrent con batches configurables
- Retry automático con backoff exponencial
- Dead letter queue para trabajos fallidos
- Telemetry integrado
- Rate limiting por tipo de trabajo

### 📡 Real-Time Events
- Phoenix PubSub para eventos en tiempo real
- Subscripción por job o canal
- Webhook delivery con retries
- Event history y replay

### 📦 Distributed Queue
- Priority queue (CRITICAL, HIGH, NORMAL, LOW, BACKGROUND)
- Distribución automática entre nodos con Horde
- Deduplicación de jobs
- Persistencia con recovery
- Métricas en tiempo real

### 🌐 Cluster Support
- Autodiscovery con libcluster
- Distributed registry con Horde
- Failover automático
- Hot code reload

## Instalación

```bash
# Requisitos: Elixir 1.15+, Erlang/OTP 26+

cd elixir_services

# Instalar dependencias
mix deps.get

# Compilar
mix compile

# Ejecutar en desarrollo
iex -S mix phx.server

# Para producción
MIX_ENV=prod mix release
_build/prod/rel/transcriber_ai/bin/transcriber_ai start
```

## Configuración

```elixir
# config/config.exs
config :transcriber_ai,
  openrouter_api_key: System.get_env("OPENROUTER_API_KEY"),
  pipeline_concurrency: 5,
  batch_size: 10,
  cluster_topology: :kubernetes

config :transcriber_ai, TranscriberAI.Repo,
  database: "transcriber_ai_#{Mix.env()}",
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  pool_size: 10

config :transcriber_ai, TranscriberAI.Cache,
  adapter: Nebulex.Adapters.Local,
  gc_interval: :timer.hours(1)
```

## Uso

### Encolar un Trabajo

```elixir
# Transcripción de video
job = %{
  type: :transcribe,
  url: "https://tiktok.com/@user/video/123",
  options: [include_timestamps: true]
}

{:ok, job_id} = TranscriberAI.Queue.DistributedQueue.enqueue(job, :high)

# Suscribirse a eventos del job
TranscriberAI.Events.Broadcaster.subscribe_to_job(job_id)

receive do
  %{type: :job_completed, payload: %{result: result}} ->
    IO.inspect(result)
end
```

### Análisis de Contenido

```elixir
job = %{
  type: :analyze,
  text: "Tu contenido aquí...",
  options: [extract_keywords: true, detect_framework: true]
}

{:ok, job_id} = TranscriberAI.Queue.DistributedQueue.enqueue(job)
```

### Generación de Variantes

```elixir
job = %{
  type: :variants,
  text: "Contenido original",
  count: 5
}

{:ok, job_id} = TranscriberAI.Queue.DistributedQueue.enqueue(job, :low)
```

### Webhooks

```elixir
# Registrar webhook
TranscriberAI.Events.Broadcaster.register_webhook(
  "https://tu-servidor.com/webhook",
  [:job_completed, :job_failed]
)

# Eventos enviados:
# - job_queued
# - job_started
# - job_completed
# - job_failed
# - job_cancelled
```

### Estadísticas de Cola

```elixir
stats = TranscriberAI.Queue.DistributedQueue.get_stats()

# %{
#   queue_lengths: %{critical: 0, high: 5, normal: 23, low: 12, background: 100},
#   processing_count: 5,
#   completed_count: 1234,
#   failed_count: 12,
#   total_processing_time: 45678
# }
```

## Arquitectura

```
elixir_services/
├── lib/
│   └── transcriber_ai/
│       ├── application.ex       # OTP Application
│       ├── pipeline/
│       │   ├── supervisor.ex    # Pipeline Supervisor
│       │   ├── producer.ex      # Broadway Producer
│       │   └── transcription_pipeline.ex
│       ├── events/
│       │   └── broadcaster.ex   # PubSub Events
│       ├── queue/
│       │   ├── supervisor.ex
│       │   └── distributed_queue.ex
│       ├── cluster/
│       │   └── supervisor.ex    # Cluster Setup
│       ├── cache.ex
│       └── repo.ex
├── config/
│   ├── config.exs
│   ├── dev.exs
│   ├── prod.exs
│   └── runtime.exs
├── mix.exs
└── README.md
```

## Cluster Topologies

### Kubernetes

```elixir
config :libcluster,
  topologies: [
    k8s: [
      strategy: Cluster.Strategy.Kubernetes,
      config: [
        kubernetes_selector: "app=transcriber-ai",
        kubernetes_node_basename: "transcriber_ai"
      ]
    ]
  ]
```

### Local Development

```elixir
config :libcluster,
  topologies: [
    local: [
      strategy: Cluster.Strategy.Epmd,
      config: [hosts: [:"node1@localhost", :"node2@localhost"]]
    ]
  ]
```

## Performance

| Operación | Throughput | Latencia P99 |
|-----------|------------|--------------|
| Enqueue | 50,000/s | <1ms |
| Process (simple) | 1,000/s | ~50ms |
| Process (transcribe) | 10/s | ~10s |
| Event broadcast | 100,000/s | <1ms |

## Monitoreo

### Phoenix LiveDashboard

Accede a `http://localhost:4000/dashboard` para:
- Métricas del sistema
- Estado de procesos
- Telemetry en tiempo real
- Logs estructurados

### Telemetry Events

```elixir
:telemetry.attach_many(
  "transcriber-metrics",
  [
    [:transcriber, :job, :success],
    [:transcriber, :job, :failure],
    [:transcriber, :batch, :processed]
  ],
  &handle_event/4,
  nil
)
```

---

**Elixir Services** - Alta disponibilidad para Social Video Transcriber AI ⚡












