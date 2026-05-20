/**
 * Funciones auxiliares para creación de modelos
 * ==============================================
 */

export interface ModelSpec {
  modelName?: string
  layers?: any[]
  optimizer?: string
  loss?: string
  metrics?: string[]
  suggestedLayers?: any[]
  estimatedComplexity?: 'low' | 'medium' | 'high'
  recommendedOptimizer?: string
  recommendedLoss?: string
  [key: string]: any
}

/**
 * Normaliza el nombre del modelo
 */
export function normalizeModelName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .substring(0, 50) || `model-${Date.now()}`
}

/**
 * Genera un nombre de modelo único basado en descripción
 */
export function generateModelNameFromDescription(description: string): string {
  const words = description
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .slice(0, 5) // Primeras 5 palabras
    .map(word => word.replace(/[^a-z0-9]/g, ''))
    .filter(word => word.length > 0)

  const baseName = words.join('-') || 'model'
  const timestamp = Date.now().toString().slice(-6)
  
  return `${baseName}-${timestamp}`
}

/**
 * Combina especificaciones de modelo
 */
export function mergeModelSpecs(
  baseSpec: ModelSpec | null | undefined,
  analysisSpec: Partial<ModelSpec>
): ModelSpec {
  const merged: ModelSpec = {
    ...baseSpec,
    ...analysisSpec
  }

  // Priorizar recomendaciones del análisis si no hay especificaciones explícitas
  if (!merged.optimizer && analysisSpec.recommendedOptimizer) {
    merged.optimizer = analysisSpec.recommendedOptimizer
  }

  if (!merged.loss && analysisSpec.recommendedLoss) {
    merged.loss = analysisSpec.recommendedLoss
  }

  if (!merged.layers && analysisSpec.suggestedLayers) {
    merged.layers = analysisSpec.suggestedLayers
  }

  return merged
}

/**
 * Valida una especificación de modelo
 */
export function validateModelSpec(spec: any): {
  valid: boolean
  errors: string[]
  warnings: string[]
} {
  const errors: string[] = []
  const warnings: string[] = []

  if (spec === null || spec === undefined) {
    return { valid: true, errors: [], warnings: [] }
  }

  if (typeof spec !== 'object' || Array.isArray(spec)) {
    errors.push('La especificación debe ser un objeto')
    return { valid: false, errors, warnings }
  }

  if (spec.modelName && typeof spec.modelName !== 'string') {
    errors.push('modelName debe ser una cadena de texto')
  }

  if (spec.layers && !Array.isArray(spec.layers)) {
    errors.push('layers debe ser un array')
  }

  if (spec.optimizer && typeof spec.optimizer !== 'string') {
    warnings.push('optimizer debe ser una cadena de texto')
  }

  if (spec.loss && typeof spec.loss !== 'string') {
    warnings.push('loss debe ser una cadena de texto')
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  }
}

/**
 * Sanitiza una especificación de modelo
 */
export function sanitizeModelSpec(spec: any): ModelSpec | null {
  if (!spec) return null

  if (typeof spec !== 'object' || Array.isArray(spec)) {
    return null
  }

  const sanitized: ModelSpec = {}

  if (spec.modelName && typeof spec.modelName === 'string') {
    sanitized.modelName = normalizeModelName(spec.modelName)
  }

  if (Array.isArray(spec.layers)) {
    sanitized.layers = spec.layers
  }

  if (typeof spec.optimizer === 'string') {
    sanitized.optimizer = spec.optimizer
  }

  if (typeof spec.loss === 'string') {
    sanitized.loss = spec.loss
  }

  if (Array.isArray(spec.metrics)) {
    sanitized.metrics = spec.metrics.filter((m: any) => typeof m === 'string')
  }

  // Preservar otros campos válidos
  Object.keys(spec).forEach(key => {
    if (!['modelName', 'layers', 'optimizer', 'loss', 'metrics'].includes(key)) {
      sanitized[key] = spec[key]
    }
  })

  return sanitized
}










