/**
 * Validadores de Tipos en Runtime
 * ================================
 * 
 * Utilidades para validar tipos en tiempo de ejecución
 */

import {
  ModelStatus,
  ModelType,
  ModelArchitecture,
  OptimizerType,
  LossFunction,
  MetricType,
  LayerType,
  ModelSpec,
  ModelState,
  ValidationResult,
  ValidationError,
  ValidationWarning
} from '../types/modelTypes'

// ============================================================================
// VALIDADORES DE TIPOS BASE
// ============================================================================

const VALID_MODEL_STATUSES: readonly ModelStatus[] = [
  'idle',
  'validating',
  'creating',
  'compiling',
  'training',
  'evaluating',
  'predicting',
  'completed',
  'failed',
  'cancelled',
  'paused'
] as const

const VALID_MODEL_TYPES: readonly ModelType[] = [
  'classification',
  'regression',
  'nlp',
  'vision',
  'time-series',
  'generative',
  'custom'
] as const

const VALID_ARCHITECTURES: readonly ModelArchitecture[] = [
  'dense',
  'cnn',
  'rnn',
  'lstm',
  'gru',
  'transformer',
  'hybrid',
  'auto'
] as const

const VALID_OPTIMIZERS: readonly OptimizerType[] = [
  'adam',
  'sgd',
  'rmsprop',
  'adagrad',
  'adamw',
  'nadam'
] as const

const VALID_LOSS_FUNCTIONS: readonly LossFunction[] = [
  'sparse_categorical_crossentropy',
  'categorical_crossentropy',
  'binary_crossentropy',
  'mean_squared_error',
  'mean_absolute_error',
  'huber',
  'cosine_similarity'
] as const

const VALID_METRICS: readonly MetricType[] = [
  'accuracy',
  'precision',
  'recall',
  'f1_score',
  'auc',
  'mae',
  'mse',
  'rmse',
  'r2_score'
] as const

const VALID_LAYER_TYPES: readonly LayerType[] = [
  'Dense',
  'Conv2D',
  'Conv1D',
  'LSTM',
  'GRU',
  'Dropout',
  'BatchNormalization',
  'MaxPooling2D',
  'AveragePooling2D',
  'Flatten',
  'Reshape',
  'Embedding',
  'Attention',
  'GlobalAveragePooling2D',
  'GlobalMaxPooling2D'
] as const

// ============================================================================
// FUNCIONES DE VALIDACIÓN
// ============================================================================

/**
 * Valida si un valor es un ModelStatus válido
 */
export function isValidModelStatus(value: unknown): value is ModelStatus {
  return typeof value === 'string' && VALID_MODEL_STATUSES.includes(value as ModelStatus)
}

/**
 * Valida si un valor es un ModelType válido
 */
export function isValidModelType(value: unknown): value is ModelType {
  return typeof value === 'string' && VALID_MODEL_TYPES.includes(value as ModelType)
}

/**
 * Valida si un valor es un ModelArchitecture válido
 */
export function isValidArchitecture(value: unknown): value is ModelArchitecture {
  return typeof value === 'string' && VALID_ARCHITECTURES.includes(value as ModelArchitecture)
}

/**
 * Valida si un valor es un OptimizerType válido
 */
export function isValidOptimizer(value: unknown): value is OptimizerType {
  return typeof value === 'string' && VALID_OPTIMIZERS.includes(value as OptimizerType)
}

/**
 * Valida si un valor es un LossFunction válido
 */
export function isValidLossFunction(value: unknown): value is LossFunction {
  return typeof value === 'string' && VALID_LOSS_FUNCTIONS.includes(value as LossFunction)
}

/**
 * Valida si un valor es un MetricType válido
 */
export function isValidMetric(value: unknown): value is MetricType {
  return typeof value === 'string' && VALID_METRICS.includes(value as MetricType)
}

/**
 * Valida si un valor es un LayerType válido
 */
export function isValidLayerType(value: unknown): value is LayerType {
  return typeof value === 'string' && VALID_LAYER_TYPES.includes(value as LayerType)
}

// ============================================================================
// VALIDADORES DE OBJETOS COMPLEJOS
// ============================================================================

/**
 * Valida un ModelSpec completo
 */
export function validateModelSpec(spec: unknown): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  if (!spec || typeof spec !== 'object') {
    errors.push({
      field: 'spec',
      message: 'La especificación del modelo debe ser un objeto',
      code: 'INVALID_SPEC_TYPE',
      severity: 'error'
    })
    return { valid: false, errors, warnings, score: 0 }
  }

  const s = spec as Record<string, unknown>

  // Validar type
  if (!s.type || !isValidModelType(s.type)) {
    errors.push({
      field: 'type',
      message: `Tipo de modelo inválido. Debe ser uno de: ${VALID_MODEL_TYPES.join(', ')}`,
      code: 'INVALID_MODEL_TYPE',
      severity: 'error'
    })
  }

  // Validar architecture
  if (!s.architecture || !isValidArchitecture(s.architecture)) {
    errors.push({
      field: 'architecture',
      message: `Arquitectura inválida. Debe ser una de: ${VALID_ARCHITECTURES.join(', ')}`,
      code: 'INVALID_ARCHITECTURE',
      severity: 'error'
    })
  }

  // Validar layers
  if (!Array.isArray(s.layers) || s.layers.length === 0) {
    errors.push({
      field: 'layers',
      message: 'El modelo debe tener al menos una capa',
      code: 'INVALID_LAYERS',
      severity: 'error'
    })
  } else {
    (s.layers as unknown[]).forEach((layer, idx) => {
      if (!layer || typeof layer !== 'object') {
        errors.push({
          field: `layers[${idx}]`,
          message: 'Cada capa debe ser un objeto',
          code: 'INVALID_LAYER_TYPE',
          severity: 'error'
        })
        return
      }

      const l = layer as Record<string, unknown>
      if (!l.type || !isValidLayerType(l.type)) {
        errors.push({
          field: `layers[${idx}].type`,
          message: `Tipo de capa inválido. Debe ser uno de: ${VALID_LAYER_TYPES.join(', ')}`,
          code: 'INVALID_LAYER_TYPE',
          severity: 'error'
        })
      }
    })
  }

  // Validar optimizer
  if (!s.optimizer || typeof s.optimizer !== 'object') {
    errors.push({
      field: 'optimizer',
      message: 'El optimizador debe ser un objeto',
      code: 'INVALID_OPTIMIZER',
      severity: 'error'
    })
  } else {
    const opt = s.optimizer as Record<string, unknown>
    if (!opt.type || !isValidOptimizer(opt.type)) {
      errors.push({
        field: 'optimizer.type',
        message: `Tipo de optimizador inválido. Debe ser uno de: ${VALID_OPTIMIZERS.join(', ')}`,
        code: 'INVALID_OPTIMIZER_TYPE',
        severity: 'error'
      })
    }
  }

  // Validar loss
  if (!s.loss || typeof s.loss !== 'object') {
    errors.push({
      field: 'loss',
      message: 'La función de pérdida debe ser un objeto',
      code: 'INVALID_LOSS',
      severity: 'error'
    })
  } else {
    const loss = s.loss as Record<string, unknown>
    if (!loss.type || !isValidLossFunction(loss.type)) {
      errors.push({
        field: 'loss.type',
        message: `Función de pérdida inválida. Debe ser una de: ${VALID_LOSS_FUNCTIONS.join(', ')}`,
        code: 'INVALID_LOSS_TYPE',
        severity: 'error'
      })
    }
  }

  // Validar metrics
  if (s.metrics) {
    if (!Array.isArray(s.metrics)) {
      errors.push({
        field: 'metrics',
        message: 'Las métricas deben ser un array',
        code: 'INVALID_METRICS_TYPE',
        severity: 'error'
      })
    } else {
      (s.metrics as unknown[]).forEach((metric, idx) => {
        if (!isValidMetric(metric)) {
          warnings.push({
            field: `metrics[${idx}]`,
            message: `Métrica inválida: ${metric}. Se ignorará.`,
            code: 'INVALID_METRIC',
            suggestion: `Usar una de: ${VALID_METRICS.join(', ')}`
          })
        }
      })
    }
  }

  // Advertencias
  if (Array.isArray(s.layers) && s.layers.length > 10) {
    warnings.push({
      field: 'layers',
      message: 'El modelo tiene muchas capas. Esto puede afectar el rendimiento.',
      code: 'TOO_MANY_LAYERS',
      suggestion: 'Considera reducir el número de capas o usar arquitecturas más eficientes'
    })
  }

  const score = errors.length === 0 
    ? (warnings.length === 0 ? 100 : Math.max(50, 100 - warnings.length * 10))
    : 0

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    score
  }
}

/**
 * Valida un ModelState
 */
export function validateModelState(state: unknown): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  if (!state || typeof state !== 'object') {
    errors.push({
      field: 'state',
      message: 'El estado del modelo debe ser un objeto',
      code: 'INVALID_STATE_TYPE',
      severity: 'error'
    })
    return { valid: false, errors, warnings, score: 0 }
  }

  const s = state as Record<string, unknown>

  // Validar id
  if (!s.id || typeof s.id !== 'string' || s.id.trim().length === 0) {
    errors.push({
      field: 'id',
      message: 'El ID del modelo es requerido y debe ser una cadena no vacía',
      code: 'INVALID_ID',
      severity: 'error'
    })
  }

  // Validar name
  if (!s.name || typeof s.name !== 'string' || s.name.trim().length === 0) {
    errors.push({
      field: 'name',
      message: 'El nombre del modelo es requerido y debe ser una cadena no vacía',
      code: 'INVALID_NAME',
      severity: 'error'
    })
  }

  // Validar status
  if (!s.status || !isValidModelStatus(s.status)) {
    errors.push({
      field: 'status',
      message: `Estado inválido. Debe ser uno de: ${VALID_MODEL_STATUSES.join(', ')}`,
      code: 'INVALID_STATUS',
      severity: 'error'
    })
  }

  // Validar progress
  if (s.progress !== undefined) {
    if (typeof s.progress !== 'number' || s.progress < 0 || s.progress > 100) {
      errors.push({
        field: 'progress',
        message: 'El progreso debe ser un número entre 0 y 100',
        code: 'INVALID_PROGRESS',
        severity: 'error'
      })
    }
  }

  // Validar dates
  if (!s.createdAt || !(s.createdAt instanceof Date)) {
    errors.push({
      field: 'createdAt',
      message: 'La fecha de creación debe ser un objeto Date',
      code: 'INVALID_CREATED_AT',
      severity: 'error'
    })
  }

  if (!s.updatedAt || !(s.updatedAt instanceof Date)) {
    errors.push({
      field: 'updatedAt',
      message: 'La fecha de actualización debe ser un objeto Date',
      code: 'INVALID_UPDATED_AT',
      severity: 'error'
    })
  }

  // Validar spec si existe
  if (s.spec) {
    const specValidation = validateModelSpec(s.spec)
    if (!specValidation.valid) {
      errors.push(...specValidation.errors.map(e => ({
        ...e,
        field: `spec.${e.field}`
      })))
    }
    warnings.push(...specValidation.warnings.map(w => ({
      ...w,
      field: `spec.${w.field}`
    })))
  }

  const score = errors.length === 0 
    ? (warnings.length === 0 ? 100 : Math.max(50, 100 - warnings.length * 10))
    : 0

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    score
  }
}

// ============================================================================
// UTILIDADES DE VALIDACIÓN
// ============================================================================

/**
 * Valida y normaliza un valor a un tipo específico
 */
export function validateAndNormalize<T>(
  value: unknown,
  validator: (v: unknown) => v is T,
  defaultValue: T,
  fieldName: string
): T {
  if (validator(value)) {
    return value
  }
  console.warn(`Valor inválido para ${fieldName}: ${value}. Usando valor por defecto: ${defaultValue}`)
  return defaultValue
}

/**
 * Valida múltiples campos de una vez
 */
export function validateFields(
  obj: Record<string, unknown>,
  validators: Record<string, (value: unknown) => boolean>
): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  for (const [field, validator] of Object.entries(validators)) {
    if (!validator(obj[field])) {
      errors.push({
        field,
        message: `Campo ${field} no es válido`,
        code: `INVALID_${field.toUpperCase()}`,
        severity: 'error'
      })
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    score: errors.length === 0 ? 100 : 0
  }
}







