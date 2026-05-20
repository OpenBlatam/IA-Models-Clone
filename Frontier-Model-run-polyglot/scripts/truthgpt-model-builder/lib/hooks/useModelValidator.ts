/**
 * Hook para validación avanzada de modelos
 * =========================================
 */

import { useCallback, useMemo } from 'react'

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
  suggestions: string[]
  score: number // 0-100
}

export interface ModelSpec {
  layers?: any[]
  optimizer?: string
  loss?: string
  metrics?: string[]
  batch_size?: number
  epochs?: number
  learning_rate?: number
  [key: string]: any
}

/**
 * Hook para validación avanzada de especificaciones de modelos
 */
export function useModelValidator() {
  const validateDescription = useCallback((description: string): ValidationResult => {
    const errors: string[] = []
    const warnings: string[] = []
    const suggestions: string[] = []

    // Validación básica
    if (!description || typeof description !== 'string') {
      return {
        valid: false,
        errors: ['La descripción debe ser una cadena de texto'],
        warnings: [],
        suggestions: [],
        score: 0
      }
    }

    const trimmed = description.trim()

    if (trimmed.length === 0) {
      errors.push('La descripción no puede estar vacía')
    } else if (trimmed.length < 10) {
      errors.push('La descripción debe tener al menos 10 caracteres')
    } else if (trimmed.length > 5000) {
      errors.push('La descripción es demasiado larga (máximo 5000 caracteres)')
    }

    // Warnings
    if (trimmed.length < 20) {
      warnings.push('Una descripción más detallada ayudará a crear un mejor modelo')
    }

    if (trimmed.length > 2000) {
      warnings.push('La descripción es muy larga. Considera simplificarla')
    }

    // Sugerencias
    const lowerDesc = trimmed.toLowerCase()
    
    if (!lowerDesc.includes('model') && !lowerDesc.includes('network') && !lowerDesc.includes('neural')) {
      suggestions.push('Considera mencionar el tipo de modelo (CNN, RNN, etc.)')
    }

    if (!lowerDesc.includes('classify') && !lowerDesc.includes('predict') && !lowerDesc.includes('regression')) {
      suggestions.push('Especifica la tarea del modelo (clasificación, predicción, etc.)')
    }

    // Calcular score
    let score = 100
    score -= errors.length * 30
    score -= warnings.length * 10
    score = Math.max(0, Math.min(100, score))

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      suggestions,
      score
    }
  }, [])

  const validateSpec = useCallback((spec: ModelSpec): ValidationResult => {
    const errors: string[] = []
    const warnings: string[] = []
    const suggestions: string[] = []

    if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
      return {
        valid: false,
        errors: ['La especificación debe ser un objeto'],
        warnings: [],
        suggestions: [],
        score: 0
      }
    }

    // Validar layers
    if (spec.layers) {
      if (!Array.isArray(spec.layers)) {
        errors.push('layers debe ser un array')
      } else if (spec.layers.length === 0) {
        errors.push('El modelo debe tener al menos una capa')
      } else if (spec.layers.length > 100) {
        warnings.push('El modelo tiene muchas capas. Esto puede ser lento de entrenar')
      }
    } else {
      warnings.push('No se especificaron capas. Se usarán capas por defecto')
    }

    // Validar optimizer
    const validOptimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw']
    if (spec.optimizer && !validOptimizers.includes(spec.optimizer.toLowerCase())) {
      warnings.push(`Optimizador "${spec.optimizer}" puede no ser reconocido. Usa uno de: ${validOptimizers.join(', ')}`)
    }

    // Validar loss
    const validLosses = [
      'sparsecategoricalcrossentropy',
      'categoricalcrossentropy',
      'binarycrossentropy',
      'meansquarederror',
      'mse',
      'meanabsoluteerror',
      'mae'
    ]
    if (spec.loss && !validLosses.includes(spec.loss.toLowerCase())) {
      warnings.push(`Función de pérdida "${spec.loss}" puede no ser reconocida`)
    }

    // Validar batch_size
    if (spec.batch_size !== undefined) {
      if (typeof spec.batch_size !== 'number' || spec.batch_size < 1) {
        errors.push('batch_size debe ser un número positivo')
      } else if (spec.batch_size > 1024) {
        warnings.push('batch_size muy grande puede causar problemas de memoria')
      } else if (spec.batch_size < 8) {
        warnings.push('batch_size muy pequeño puede hacer el entrenamiento lento')
      }
    }

    // Validar epochs
    if (spec.epochs !== undefined) {
      if (typeof spec.epochs !== 'number' || spec.epochs < 1) {
        errors.push('epochs debe ser un número positivo')
      } else if (spec.epochs > 1000) {
        warnings.push('epochs muy alto puede causar overfitting')
      }
    }

    // Validar learning_rate
    if (spec.learning_rate !== undefined) {
      if (typeof spec.learning_rate !== 'number' || spec.learning_rate <= 0) {
        errors.push('learning_rate debe ser un número positivo')
      } else if (spec.learning_rate > 1) {
        warnings.push('learning_rate muy alto puede causar inestabilidad')
      } else if (spec.learning_rate < 0.00001) {
        warnings.push('learning_rate muy bajo puede hacer el entrenamiento muy lento')
      }
    }

    // Sugerencias
    if (!spec.batch_size) {
      suggestions.push('Considera especificar batch_size (32 es un buen valor por defecto)')
    }

    if (!spec.epochs) {
      suggestions.push('Considera especificar epochs (10-100 es un rango común)')
    }

    if (!spec.learning_rate) {
      suggestions.push('Considera especificar learning_rate (0.001 es un buen valor por defecto)')
    }

    // Calcular score
    let score = 100
    score -= errors.length * 30
    score -= warnings.length * 5
    score = Math.max(0, Math.min(100, score))

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      suggestions,
      score
    }
  }, [])

  const validateComplete = useCallback((
    description: string,
    spec?: ModelSpec
  ): ValidationResult => {
    const descResult = validateDescription(description)
    const specResult = spec ? validateSpec(spec) : {
      valid: true,
      errors: [],
      warnings: [],
      suggestions: [],
      score: 100
    }

    return {
      valid: descResult.valid && specResult.valid,
      errors: [...descResult.errors, ...specResult.errors],
      warnings: [...descResult.warnings, ...specResult.warnings],
      suggestions: [...descResult.suggestions, ...specResult.suggestions],
      score: (descResult.score + specResult.score) / 2
    }
  }, [validateDescription, validateSpec])

  return {
    validateDescription,
    validateSpec,
    validateComplete
  }
}

