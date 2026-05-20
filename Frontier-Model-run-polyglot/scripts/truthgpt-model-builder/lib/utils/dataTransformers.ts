/**
 * Utilidades de Transformación de Datos
 * ======================================
 * 
 * Funciones para transformar datos entre diferentes formatos
 */

import {
  ModelSpec,
  ModelState,
  ModelProgress,
  ModelStatus,
  LayerConfig,
  OptimizerConfig,
  LossConfig
} from '../types/modelTypes'

// ============================================================================
// TRANSFORMACIONES DE MODEL SPEC
// ============================================================================

/**
 * Convierte un ModelSpec a formato de API
 */
export function transformSpecToAPI(spec: ModelSpec): Record<string, unknown> {
  return {
    type: spec.type,
    architecture: spec.architecture,
    layers: spec.layers.map(layer => ({
      type: layer.type,
      ...layer.params
    })),
    input_shape: spec.inputShape,
    output_shape: spec.outputShape,
    optimizer: {
      type: spec.optimizer.type,
      learning_rate: spec.optimizer.learningRate,
      ...spec.optimizer.params
    },
    loss: {
      type: spec.loss.type,
      ...spec.loss.params
    },
    metrics: spec.metrics,
    training: spec.training ? {
      epochs: spec.training.epochs,
      batch_size: spec.training.batchSize,
      validation_split: spec.training.validationSplit,
      shuffle: spec.training.shuffle,
      verbose: spec.training.verbose
    } : undefined,
    compilation: spec.compilation
  }
}

/**
 * Convierte formato de API a ModelSpec
 */
export function transformAPIToSpec(apiData: Record<string, unknown>): ModelSpec {
  const layers = (apiData.layers as unknown[] || []).map((layer: unknown) => {
    const l = layer as Record<string, unknown>
    const { type, ...params } = l
    return {
      type: type as string,
      params: params as Record<string, unknown>
    } as LayerConfig
  })

  const optimizer = apiData.optimizer as Record<string, unknown>
  const loss = apiData.loss as Record<string, unknown>

  return {
    type: apiData.type as ModelSpec['type'],
    architecture: apiData.architecture as ModelSpec['architecture'],
    layers,
    inputShape: apiData.input_shape as number[] | undefined,
    outputShape: apiData.output_shape as number[] | undefined,
    optimizer: {
      type: optimizer.type as OptimizerConfig['type'],
      learningRate: optimizer.learning_rate as number | undefined,
      params: { ...optimizer } as Record<string, unknown>
    },
    loss: {
      type: loss.type as LossConfig['type'],
      params: { ...loss } as Record<string, unknown>
    },
    metrics: (apiData.metrics as string[]) || [],
    training: apiData.training ? {
      epochs: (apiData.training as Record<string, unknown>).epochs as number,
      batchSize: (apiData.training as Record<string, unknown>).batch_size as number,
      validationSplit: (apiData.training as Record<string, unknown>).validation_split as number | undefined,
      shuffle: (apiData.training as Record<string, unknown>).shuffle as boolean | undefined,
      verbose: (apiData.training as Record<string, unknown>).verbose as number | undefined
    } : undefined,
    compilation: apiData.compilation as ModelSpec['compilation']
  }
}

// ============================================================================
// TRANSFORMACIONES DE MODEL STATE
// ============================================================================

/**
 * Convierte un ModelState a formato legible
 */
export function transformStateToReadable(state: ModelState): {
  id: string
  name: string
  status: string
  progress: string
  createdAt: string
  updatedAt: string
  error?: string
} {
  const statusLabels: Record<ModelStatus, string> = {
    idle: 'Inactivo',
    validating: 'Validando',
    creating: 'Creando',
    compiling: 'Compilando',
    training: 'Entrenando',
    evaluating: 'Evaluando',
    predicting: 'Prediciendo',
    completed: 'Completado',
    failed: 'Fallido',
    cancelled: 'Cancelado',
    paused: 'Pausado'
  }

  return {
    id: state.id,
    name: state.name,
    status: statusLabels[state.status] || state.status,
    progress: `${state.progress}%`,
    createdAt: state.createdAt.toLocaleString(),
    updatedAt: state.updatedAt.toLocaleString(),
    error: state.error ? `${state.error.code}: ${state.error.message}` : undefined
  }
}

/**
 * Crea un ModelState desde datos parciales
 */
export function createModelState(
  partial: Partial<ModelState> & { id: string; name: string }
): ModelState {
  const now = new Date()
  
  return {
    id: partial.id,
    name: partial.name,
    status: partial.status || 'idle',
    progress: partial.progress ?? 0,
    currentStep: partial.currentStep,
    error: partial.error,
    createdAt: partial.createdAt || now,
    updatedAt: partial.updatedAt || now,
    completedAt: partial.completedAt,
    spec: partial.spec,
    metadata: partial.metadata
  }
}

// ============================================================================
// TRANSFORMACIONES DE PROGRESO
// ============================================================================

/**
 * Actualiza el progreso de un modelo
 */
export function updateProgress(
  current: ModelProgress,
  updates: Partial<Pick<ModelProgress, 'step' | 'progress' | 'message' | 'estimatedTimeRemaining'>>
): ModelProgress {
  return {
    ...current,
    ...updates,
    updatedAt: new Date()
  }
}

/**
 * Calcula el tiempo estimado restante basado en el progreso
 */
export function calculateEstimatedTime(
  progress: ModelProgress,
  averageTimePerStep?: number
): number | undefined {
  if (!averageTimePerStep) return undefined

  const elapsed = (Date.now() - progress.startedAt.getTime()) / 1000 // segundos
  const progressRatio = progress.progress / 100
  const totalEstimated = elapsed / progressRatio
  const remaining = totalEstimated - elapsed

  return remaining > 0 ? remaining : undefined
}

// ============================================================================
// TRANSFORMACIONES DE CAPAS
// ============================================================================

/**
 * Normaliza una configuración de capa
 */
export function normalizeLayerConfig(layer: unknown): LayerConfig | null {
  if (!layer || typeof layer !== 'object') return null

  const l = layer as Record<string, unknown>
  
  if (!l.type || typeof l.type !== 'string') return null

  return {
    type: l.type as LayerConfig['type'],
    params: (l.params as Record<string, unknown>) || {},
    name: l.name as string | undefined,
    activation: l.activation as string | undefined
  }
}

/**
 * Valida y normaliza un array de capas
 */
export function normalizeLayers(layers: unknown): LayerConfig[] {
  if (!Array.isArray(layers)) return []

  return layers
    .map(normalizeLayerConfig)
    .filter((layer): layer is LayerConfig => layer !== null)
}

// ============================================================================
// TRANSFORMACIONES DE OPTIMIZADOR Y LOSS
// ============================================================================

/**
 * Normaliza una configuración de optimizador
 */
export function normalizeOptimizerConfig(optimizer: unknown): OptimizerConfig | null {
  if (!optimizer || typeof optimizer !== 'object') return null

  const opt = optimizer as Record<string, unknown>
  
  if (!opt.type || typeof opt.type !== 'string') return null

  return {
    type: opt.type as OptimizerConfig['type'],
    learningRate: opt.learningRate || opt.learning_rate as number | undefined,
    params: { ...opt } as Record<string, unknown>
  }
}

/**
 * Normaliza una configuración de loss
 */
export function normalizeLossConfig(loss: unknown): LossConfig | null {
  if (!loss || typeof loss !== 'object') return null

  const l = loss as Record<string, unknown>
  
  if (!l.type || typeof l.type !== 'string') return null

  return {
    type: l.type as LossConfig['type'],
    params: { ...l } as Record<string, unknown>
  }
}

// ============================================================================
// TRANSFORMACIONES DE ERRORES
// ============================================================================

/**
 * Convierte un error a formato legible
 */
export function transformErrorToReadable(error: unknown): {
  code: string
  message: string
  details?: string
} {
  if (error instanceof Error) {
    return {
      code: 'UNKNOWN_ERROR',
      message: error.message,
      details: error.stack
    }
  }

  if (error && typeof error === 'object' && 'code' in error && 'message' in error) {
    return {
      code: String(error.code),
      message: String(error.message),
      details: 'details' in error ? String(error.details) : undefined
    }
  }

  return {
    code: 'UNKNOWN_ERROR',
    message: String(error || 'Error desconocido')
  }
}

// ============================================================================
// UTILIDADES DE SERIALIZACIÓN
// ============================================================================

/**
 * Serializa un ModelState a JSON seguro
 */
export function serializeModelState(state: ModelState): string {
  return JSON.stringify({
    ...state,
    createdAt: state.createdAt.toISOString(),
    updatedAt: state.updatedAt.toISOString(),
    completedAt: state.completedAt?.toISOString(),
    error: state.error ? {
      ...state.error,
      timestamp: state.error.timestamp.toISOString()
    } : undefined
  })
}

/**
 * Deserializa un ModelState desde JSON
 */
export function deserializeModelState(json: string): ModelState | null {
  try {
    const data = JSON.parse(json)
    return {
      ...data,
      createdAt: new Date(data.createdAt),
      updatedAt: new Date(data.updatedAt),
      completedAt: data.completedAt ? new Date(data.completedAt) : undefined,
      error: data.error ? {
        ...data.error,
        timestamp: new Date(data.error.timestamp)
      } : undefined
    } as ModelState
  } catch {
    return null
  }
}







