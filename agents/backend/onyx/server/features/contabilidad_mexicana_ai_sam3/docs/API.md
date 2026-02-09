# API Documentation

## REST API Endpoints

### POST /calcular-impuestos

Calculate taxes for a fiscal regime.

**Request Body:**
```json
{
  "regimen": "RESICO",
  "tipo_impuesto": "ISR",
  "datos": {
    "ingresos": 100000,
    "gastos": 30000,
    "periodo": "2024-01"
  },
  "priority": 0
}
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "submitted"
}
```

### POST /asesoria-fiscal

Get fiscal advice.

**Request Body:**
```json
{
  "pregunta": "¿Puedo deducir gastos de home office?",
  "contexto": {
    "regimen": "RESICO",
    "ingresos_anuales": 500000
  },
  "priority": 0
}
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "submitted"
}
```

### POST /guia-fiscal

Get fiscal guide.

**Request Body:**
```json
{
  "tema": "Deducciones RESICO",
  "nivel_detalle": "completo",
  "priority": 0
}
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "submitted"
}
```

### POST /tramite-sat

Get SAT procedure information.

**Request Body:**
```json
{
  "tipo_tramite": "Alta en RFC",
  "detalles": {
    "persona_fisica": true
  },
  "priority": 0
}
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "submitted"
}
```

### POST /ayuda-declaracion

Get declaration assistance.

**Request Body:**
```json
{
  "tipo_declaracion": "mensual",
  "periodo": "2024-01",
  "datos": {
    "rfc": "ABC123456789"
  },
  "priority": 0
}
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "submitted"
}
```

### GET /task/{task_id}/status

Get task status.

**Response:**
```json
{
  "id": "uuid-here",
  "service_type": "calcular_impuestos",
  "status": "completed",
  "created_at": "2024-01-01T00:00:00",
  "started_at": "2024-01-01T00:00:01",
  "completed_at": "2024-01-01T00:00:05",
  "priority": 0
}
```

### GET /task/{task_id}/result

Get task result.

**Response:**
```json
{
  "resultado": "ISR calculado: $5,000",
  "tokens_used": 100,
  "model": "anthropic/claude-3.5-sonnet",
  "tiempo_calculo": "2024-01-01T00:00:05"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "agent_running": true
}
```

## Python API

### ContadorSAM3Agent

Main agent class for processing accounting tasks.

#### Methods

##### calcular_impuestos(regimen, tipo_impuesto, datos, priority=0)

Calculate taxes.

**Parameters:**
- `regimen` (str): Fiscal regime
- `tipo_impuesto` (str): Tax type
- `datos` (dict): Calculation data
- `priority` (int): Task priority

**Returns:**
- `str`: Task ID

##### asesoria_fiscal(pregunta, contexto=None, priority=0)

Get fiscal advice.

**Parameters:**
- `pregunta` (str): Question
- `contexto` (dict, optional): Additional context
- `priority` (int): Task priority

**Returns:**
- `str`: Task ID

##### guia_fiscal(tema, nivel_detalle="completo", priority=0)

Get fiscal guide.

**Parameters:**
- `tema` (str): Topic
- `nivel_detalle` (str): Detail level
- `priority` (int): Task priority

**Returns:**
- `str`: Task ID

##### tramite_sat(tipo_tramite, detalles=None, priority=0)

Get SAT procedure information.

**Parameters:**
- `tipo_tramite` (str): Procedure type
- `detalles` (dict, optional): Additional details
- `priority` (int): Task priority

**Returns:**
- `str`: Task ID

##### ayuda_declaracion(tipo_declaracion, periodo, datos=None, priority=0)

Get declaration assistance.

**Parameters:**
- `tipo_declaracion` (str): Declaration type
- `periodo` (str): Fiscal period
- `datos` (dict, optional): Taxpayer data
- `priority` (int): Task priority

**Returns:**
- `str`: Task ID

##### get_task_status(task_id)

Get task status.

**Parameters:**
- `task_id` (str): Task ID

**Returns:**
- `dict`: Task status

##### get_task_result(task_id)

Get task result.

**Parameters:**
- `task_id` (str): Task ID

**Returns:**
- `dict`: Task result or None

##### start()

Start agent in continuous mode.

##### stop()

Stop agent.

##### close()

Close agent and cleanup resources.
