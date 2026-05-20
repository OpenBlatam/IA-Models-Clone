/**
 * Hook optimizado para creación de modelos con caché y validación mejorada
 * ========================================================================
 */

import { useState, useCallback, useMemo } from 'react'
import { validateDescription } from '../validator'
import { analyzeModelDescription } from '../modules/management'

export interface OptimizedModelCreationOptions {
  description: string
  spec?: any
  enableCache?: boolean
  enableValidation?: boolean
  enableAnalysis?: boolean
}

export interface OptimizedModelCreationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
  analysis?: {
    suggestedLayers?: any[]
    estimatedComplexity?: 'low' | 'medium' | 'high'
    recommendedOptimizer?: string
    recommendedLoss?: string
  }
  normalizedDescription: string
}

/**
 * Hook para creación optimizada de modelos con validación y análisis previo
 */
export function useOptimizedModelCreation() {
  const [cache, setCache] = useState<Map<string, OptimizedModelCreationResult>>(new Map())

  const prepareModelCreation = useCallback(
    async (options: OptimizedModelCreationOptions): Promise<OptimizedModelCreationResult> => {
      const {
        description,
        enableCache = true,
        enableValidation = true,
        enableAnalysis = true,
      } = options

      // Normalizar descripción
      const normalizedDescription = description.trim().toLowerCase()

      // Verificar caché
      if (enableCache && cache.has(normalizedDescription)) {
        const cached = cache.get(normalizedDescription)!
        return {
          ...cached,
          normalizedDescription: description.trim(), // Mantener formato original
        }
      }

      const result: OptimizedModelCreationResult = {
        isValid: true,
        errors: [],
        warnings: [],
        normalizedDescription: description.trim(),
      }

      // Validación
      if (enableValidation) {
        const validation = validateDescription(description)
        if (!validation.valid) {
          result.isValid = false
          result.errors.push(validation.error || 'Descripción inválida')
        } else if (validation.warnings) {
          result.warnings.push(...validation.warnings)
        }
      }

      // Análisis (solo si es válido)
      if (enableAnalysis && result.isValid) {
        try {
          const analysis = analyzeModelDescription(description)
          
          result.analysis = {
            suggestedLayers: analysis.layers,
            estimatedComplexity: analysis.complexity || 'medium',
            recommendedOptimizer: analysis.optimizer || 'adam',
            recommendedLoss: analysis.loss || 'sparsecategoricalcrossentropy',
          }
        } catch (error) {
          console.warn('Error analyzing model description:', error)
          result.warnings.push('No se pudo analizar la descripción del modelo')
        }
      }

      // Guardar en caché
      if (enableCache && result.isValid) {
        setCache(prev => {
          const next = new Map(prev)
          next.set(normalizedDescription, result)
          return next
        })
      }

      return result
    },
    [cache]
  )

  const clearCache = useCallback(() => {
    setCache(new Map())
  }, [])

  const cacheStats = useMemo(() => {
    return {
      size: cache.size,
      entries: Array.from(cache.keys()),
    }
  }, [cache])

  return {
    prepareModelCreation,
    clearCache,
    cacheStats,
  }
}

