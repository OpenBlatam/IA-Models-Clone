# 📊 Diagramas - Blatam Academy Features

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                      NGINX (Load Balancer)                       │
│                         Port 80/443                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        ┌───────▼────────┐      ┌────────▼────────┐
        │ Integration     │      │  Other Services │
        │ System          │      │                 │
        │ :8000           │      │  :8003-8005     │
        └───────┬────────┘      └────────────────┘
                │
        ┌───────▼────────┐
        │   BUL System   │
        │   :8002        │
        │                │
        │  ┌──────────┐  │
        │  │ KV Cache │  │
        │  │  Engine  │  │
        │  └────┬─────┘  │
        └───────┼────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼───┐  ┌───▼───┐  ┌───▼───┐
│PostgreSQL│ │ Redis │ │  GPU  │
│  :5432  │ │ :6379 │ │       │
└─────────┘ └───────┘ └───────┘
```

## ⚡ Ultra Adaptive KV Cache Engine - Flujo de Datos

```
Request Input
    │
    ▼
┌─────────────────────┐
│  Request Validator  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Cache Lookup       │
│  (Key Hash)         │
└──────────┬──────────┘
           │
      ┌────┴────┐
      │         │
  Cache Hit  Cache Miss
      │         │
      │         ▼
      │    ┌──────────────┐
      │    │ Process      │
      │    │ (TruthGPT)   │
      │    └──────┬───────┘
      │           │
      └───────┬───┘
              │
              ▼
      ┌──────────────┐
      │ Store in     │
      │ Cache        │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │ Compression  │
      │ (if enabled) │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │ Persistence  │
      │ (if enabled) │
      └──────┬───────┘
             │
             ▼
         Response
```

## 🔄 Estrategias de Cache

### LRU (Least Recently Used)
```
┌─────────────────────────────────────┐
│  Cache State (Most Recent → Oldest) │
├─────────────────────────────────────┤
│  [New] → [Entry 1] → [Entry 2] →    │
│  [Entry 3] → [Entry 4] → [Old]      │
└─────────────────────────────────────┘
          When Full: Evict → [Old]
```

### LFU (Least Frequently Used)
```
┌─────────────────────────────────────┐
│  Access Frequency Count              │
├─────────────────────────────────────┤
│  Entry A: ████████ (8 accesses)     │
│  Entry B: ████ (4 accesses)         │
│  Entry C: ██ (2 accesses) ← Evict  │
└─────────────────────────────────────┘
```

### Adaptive Strategy
```
┌─────────────────────────────────────┐
│  Monitoring → Pattern Detection     │
├─────────────────────────────────────┤
│  Workload Analysis                  │
│    │                                │
│    ├─ Sequential → Use LRU         │
│    ├─ Random → Use LFU             │
│    └─ Mixed → Use Adaptive          │
│                                    │
│  Auto-adjust every N requests       │
└─────────────────────────────────────┘
```

## 🚀 Pipeline de Procesamiento BUL

```
┌──────────────┐
│   Query      │
│  Input       │
└──────┬───────┘
       │
       ▼
┌──────────────┐      ┌──────────────┐
│  Priority    │      │  Rate        │
│  Queue        │      │  Limiter     │
└──────┬───────┘      └──────────────┘
       │
       ▼
┌──────────────┐
│  KV Cache    │ ◄─────┐
│  Check       │        │
└──────┬───────┘        │
       │                 │
   ┌───┴───┐             │
   │       │             │
 Hit      Miss          │
   │       │             │
   │       ▼             │
   │  ┌──────────────┐  │
   │  │ TruthGPT     │  │
   │  │ Processing   │  │
   │  └──────┬───────┘  │
   │         │          │
   └─────────┼──────────┘
             │
             ▼
      ┌──────────────┐
      │  Store in    │
      │  Cache       │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │  Document     │
      │  Generation   │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │  Response     │
      └──────────────┘
```

## 💾 Persistencia y Backup

```
┌─────────────────────────────────────┐
│  In-Memory Cache                    │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │Entry│ │Entry│ │Entry│ ...       │
│  └─────┘ └─────┘ └─────┘           │
└──────┬──────────────────────────────┘
       │
       │ Serialization
       ▼
┌─────────────────────────────────────┐
│  Disk Cache                         │
│  /data/cache/cache.pt               │
│  ├── metadata.json                  │
│  ├── entries/                       │
│  │   ├── entry_001.pt              │
│  │   ├── entry_002.pt              │
│  │   └── ...                        │
└─────────────────────────────────────┘
       │
       │ Backup
       ▼
┌─────────────────────────────────────┐
│  Backup Storage                      │
│  /backup/cache_20240101/            │
│  └── cache_backup.pt                │
└─────────────────────────────────────┘
```

## 🎯 Multi-GPU Load Balancing

```
┌─────────────────────────────────────┐
│  Request Queue                       │
└──────┬───────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Load Balancer                       │
│  ┌─────────────────────────────┐    │
│  │  GPU Utilization Monitor    │    │
│  └─────────────────────────────┘    │
└──────┬───────────────────────────────┘
       │
   ┌───┴───┬─────────┬─────────┐
   │       │         │         │
   ▼       ▼         ▼         ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│
│ 30% │ │ 45% │ │ 20% │ │ 15% │
└─────┘ └─────┘ └─────┘ └─────┘
```

## 📊 Sistema de Monitoreo

```
┌─────────────────────────────────────┐
│  KV Cache Engine                    │
│  ┌───────────────────────────────┐ │
│  │  Metrics Collector            │ │
│  │  - Latency (P50, P95, P99)   │ │
│  │  - Throughput                 │ │
│  │  - Cache Hit Rate             │ │
│  │  - Memory Usage               │ │
│  │  - GPU Utilization            │ │
│  └───────────────┬───────────────┘ │
└──────────────────┼──────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   ┌──────────┐     ┌──────────┐
   │Prometheus│     │  Grafana │
   │ :9090    │     │  :3000   │
   └──────────┘     └──────────┘
```

## 🔒 Flujo de Seguridad

```
Request
  │
  ▼
┌──────────────┐
│  HMAC        │
│  Validation  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Rate        │
│  Limiting    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Input       │
│  Sanitization│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Access      │
│  Control     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Processing  │
└──────────────┘
```

## 🔄 Flujo de Actualización/Migración

```
Current Version (v1.x)
    │
    ▼
┌──────────────┐
│  Backup      │
│  Everything  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Update      │
│  Code        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Migrate     │
│  Config      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Migrate     │
│  Data        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Verify      │
│  & Test      │
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
Success  Failure
   │       │
   │       ▼
   │   ┌──────────────┐
   │   │  Rollback    │
   │   └──────────────┘
   │
   ▼
New Version (v2.0)
```

## 📈 Escalabilidad Horizontal

```
                    Load Balancer
                         │
          ┌──────────────┼──────────────┐
          │              │              │
      ┌───▼───┐      ┌───▼───┐      ┌───▼───┐
      │ BUL   │      │ BUL   │      │ BUL   │
      │ Node 1│      │ Node 2│      │ Node 3│
      └───┬───┘      └───┬───┘      └───┬───┘
          │              │              │
          └──────────────┼──────────────┘
                         │
              ┌──────────┴──────────┐
              │                       │
         ┌────▼────┐           ┌────▼────┐
         │Shared   │           │Shared   │
         │Database │           │Redis    │
         └─────────┘           └─────────┘
```

---

**Más información:**
- [Guía de Arquitectura](ARCHITECTURE_GUIDE.md)
- [README Principal](README.md)



