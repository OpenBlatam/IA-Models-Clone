/**
 * Utilidades generales para modelos
 * ===================================
 */

/**
 * Formatea un número de parámetros
 */
export function formatParameterCount(count: number): string {
  if (count >= 1000000) {
    return `${(count / 1000000).toFixed(2)}M`
  }
  if (count >= 1000) {
    return `${(count / 1000).toFixed(2)}K`
  }
  return count.toString()
}

/**
 * Formatea un tamaño de memoria
 */
export function formatMemorySize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = bytes
  let unitIndex = 0

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }

  return `${size.toFixed(2)} ${units[unitIndex]}`
}

/**
 * Formatea un tiempo
 */
export function formatTime(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`
  }
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`
  }
  if (ms < 3600000) {
    return `${(ms / 60000).toFixed(1)}min`
  }
  return `${(ms / 3600000).toFixed(1)}h`
}

/**
 * Formatea un porcentaje
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`
}

/**
 * Calcula el tamaño estimado de un modelo
 */
export function estimateModelSize(layers: any[]): number {
  let totalParams = 0

  layers.forEach(layer => {
    const params = layer.params || {}
    
    if (layer.type === 'dense') {
      const inputSize = params.input_shape?.[0] || 128
      const units = params.units || 64
      totalParams += inputSize * units + units // weights + bias
    } else if (layer.type === 'conv2d') {
      const filters = params.filters || 32
      const kernelSize = params.kernel_size || 3
      const inputChannels = params.input_shape?.[2] || 3
      totalParams += kernelSize * kernelSize * inputChannels * filters + filters
    } else if (layer.type === 'lstm' || layer.type === 'gru') {
      const units = params.units || 64
      const inputSize = params.input_shape?.[1] || 128
      // LSTM: 4 * (input_size * units + units * units + units)
      totalParams += 4 * (inputSize * units + units * units + units)
    }
  })

  // Estimación: 4 bytes por parámetro (float32)
  return totalParams * 4
}

/**
 * Genera un nombre único para un modelo
 */
export function generateUniqueModelName(baseName: string, existingNames: string[]): string {
  let name = baseName
  let counter = 1

  while (existingNames.includes(name)) {
    name = `${baseName}-${counter}`
    counter++
  }

  return name
}

/**
 * Valida un nombre de modelo
 */
export function validateModelName(name: string): {
  valid: boolean
  error?: string
} {
  if (!name || typeof name !== 'string') {
    return { valid: false, error: 'El nombre debe ser una cadena de texto' }
  }

  const trimmed = name.trim()

  if (trimmed.length === 0) {
    return { valid: false, error: 'El nombre no puede estar vacío' }
  }

  if (trimmed.length < 3) {
    return { valid: false, error: 'El nombre debe tener al menos 3 caracteres' }
  }

  if (trimmed.length > 100) {
    return { valid: false, error: 'El nombre es demasiado largo (máximo 100 caracteres)' }
  }

  if (!/^[a-zA-Z0-9_-]+$/.test(trimmed)) {
    return { valid: false, error: 'El nombre solo puede contener letras, números, guiones y guiones bajos' }
  }

  return { valid: true }
}

/**
 * Extrae tags de una descripción
 */
export function extractTags(description: string): string[] {
  const tags: string[] = []
  const lowerDesc = description.toLowerCase()

  // Tags comunes
  const commonTags = [
    'cnn', 'lstm', 'gru', 'rnn', 'transformer', 'attention',
    'classification', 'regression', 'clustering', 'detection',
    'image', 'text', 'audio', 'video', 'time-series',
    'deep', 'shallow', 'simple', 'complex'
  ]

  commonTags.forEach(tag => {
    if (lowerDesc.includes(tag)) {
      tags.push(tag)
    }
  })

  return tags
}

/**
 * Calcula la complejidad de una descripción
 */
export function calculateComplexity(description: string): {
  score: number
  level: 'low' | 'medium' | 'high'
  factors: string[]
} {
  const factors: string[] = []
  let score = 0
  const lowerDesc = description.toLowerCase()

  // Factores que aumentan complejidad
  if (lowerDesc.includes('deep') || lowerDesc.includes('many layers')) {
    score += 30
    factors.push('Múltiples capas')
  }

  if (lowerDesc.includes('attention') || lowerDesc.includes('transformer')) {
    score += 25
    factors.push('Arquitectura avanzada')
  }

  if (lowerDesc.includes('residual') || lowerDesc.includes('skip connection')) {
    score += 20
    factors.push('Conexiones residuales')
  }

  if (lowerDesc.includes('cnn') && lowerDesc.includes('lstm')) {
    score += 15
    factors.push('Arquitectura híbrida')
  }

  if (lowerDesc.includes('large') || lowerDesc.includes('big')) {
    score += 10
    factors.push('Datos grandes')
  }

  // Factores que reducen complejidad
  if (lowerDesc.includes('simple') || lowerDesc.includes('basic')) {
    score -= 10
    factors.push('Arquitectura simple')
  }

  score = Math.max(0, Math.min(100, score))

  let level: 'low' | 'medium' | 'high'
  if (score < 30) {
    level = 'low'
  } else if (score < 70) {
    level = 'medium'
  } else {
    level = 'high'
  }

  return { score, level, factors }
}










