/**
 * Enhanced Adaptive Analyzer
 * Mejoras en el análisis adaptativo
 */

import { ModelSpec } from '@/lib/adaptive-analyzer'

export interface EnhancedAnalysis {
  spec: ModelSpec
  confidence: number
  suggestions: string[]
  warnings: string[]
  estimatedDuration: number
  complexity: 'low' | 'medium' | 'high'
  resourceRequirements: {
    gpu: boolean
    memory: string
    storage: string
  }
}

export class EnhancedAdaptiveAnalyzer {
  private contextHistory: string[] = []
  private maxHistorySize = 20

  /**
   * Analizar descripción con mejoras
   */
  analyze(description: string, context?: string[]): EnhancedAnalysis {
    // Agregar al historial de contexto
    if (context) {
      this.contextHistory.push(...context)
      if (this.contextHistory.length > this.maxHistorySize) {
        this.contextHistory.shift()
      }
    }

    // Análisis básico
    const spec = this.generateBasicSpec(description)
    
    // Mejoras inteligentes
    const confidence = this.calculateConfidence(description, spec)
    const suggestions = this.generateSuggestions(description, spec)
    const warnings = this.generateWarnings(description, spec)
    const estimatedDuration = this.estimateDuration(spec)
    const complexity = this.assessComplexity(spec)
    const resourceRequirements = this.estimateResources(spec)

    return {
      spec,
      confidence,
      suggestions,
      warnings,
      estimatedDuration,
      complexity,
      resourceRequirements,
    }
  }

  /**
   * Generar especificación básica
   */
  private generateBasicSpec(description: string): ModelSpec {
    const descLower = description.toLowerCase()

    // Detectar tipo de tarea
    let architecture = 'transformer'
    if (descLower.includes('clasificación') || descLower.includes('clasificar')) {
      architecture = 'classifier'
    } else if (descLower.includes('generación') || descLower.includes('generar')) {
      architecture = 'generator'
    } else if (descLower.includes('análisis') || descLower.includes('sentimiento')) {
      architecture = 'analyzer'
    }

    // Detectar parámetros
    const learningRate = this.extractLearningRate(description) || 0.001
    const batchSize = this.extractBatchSize(description) || 32
    const epochs = this.extractEpochs(description) || 10

    return {
      architecture,
      parameters: {
        learningRate,
        batchSize,
        epochs,
      },
      training: {
        learningRate,
        batchSize,
        epochs,
        optimizer: 'adamw',
      },
      data: {
        dataset: 'custom',
        preprocessing: ['normalize', 'tokenize'],
      },
    }
  }

  /**
   * Extraer learning rate
   */
  private extractLearningRate(description: string): number | null {
    const match = description.match(/(?:learning rate|lr|learning_rate)[\s:=]+(\d+\.?\d*)/i)
    if (match) {
      const value = parseFloat(match[1])
      if (value > 0 && value < 1) return value
    }
    return null
  }

  /**
   * Extraer batch size
   */
  private extractBatchSize(description: string): number | null {
    const match = description.match(/(?:batch size|batch_size|batch)[\s:=]+(\d+)/i)
    if (match) {
      const value = parseInt(match[1])
      if (value > 0 && value <= 1024) return value
    }
    return null
  }

  /**
   * Extraer epochs
   */
  private extractEpochs(description: string): number | null {
    const match = description.match(/(?:epochs|epoch)[\s:=]+(\d+)/i)
    if (match) {
      const value = parseInt(match[1])
      if (value > 0 && value <= 1000) return value
    }
    return null
  }

  /**
   * Calcular confianza
   */
  private calculateConfidence(description: string, spec: ModelSpec): number {
    let confidence = 0.5

    // Más confianza si hay detalles específicos
    if (description.length > 50) confidence += 0.1
    if (description.length > 100) confidence += 0.1
    if (spec.parameters?.learningRate) confidence += 0.1
    if (spec.parameters?.batchSize) confidence += 0.1
    if (spec.parameters?.epochs) confidence += 0.1

    // Más confianza si hay contexto histórico
    if (this.contextHistory.length > 0) confidence += 0.1

    return Math.min(confidence, 1.0)
  }

  /**
   * Generar sugerencias
   */
  private generateSuggestions(description: string, spec: ModelSpec): string[] {
    const suggestions: string[] = []

    if (!spec.parameters?.learningRate) {
      suggestions.push('Considera especificar un learning rate (recomendado: 0.001)')
    }

    if (!spec.parameters?.batchSize) {
      suggestions.push('Considera especificar un batch size (recomendado: 32)')
    }

    if (description.length < 50) {
      suggestions.push('Agrega más detalles sobre el propósito del modelo para mejores resultados')
    }

    if (spec.architecture === 'transformer' && !description.includes('attention')) {
      suggestions.push('Para modelos transformer, considera ajustar la configuración de atención')
    }

    return suggestions
  }

  /**
   * Generar advertencias
   */
  private generateWarnings(description: string, spec: ModelSpec): string[] {
    const warnings: string[] = []

    if (spec.parameters?.learningRate && spec.parameters.learningRate > 0.1) {
      warnings.push('Learning rate muy alto puede causar inestabilidad en el entrenamiento')
    }

    if (spec.parameters?.batchSize && spec.parameters.batchSize > 256) {
      warnings.push('Batch size muy grande puede requerir mucha memoria')
    }

    if (spec.parameters?.epochs && spec.parameters.epochs > 100) {
      warnings.push('Muchas epochs pueden llevar mucho tiempo de entrenamiento')
    }

    return warnings
  }

  /**
   * Estimar duración
   */
  private estimateDuration(spec: ModelSpec): number {
    let duration = 30000 // 30 segundos base

    // Ajustar por arquitectura
    if (spec.architecture === 'transformer') duration *= 2
    if (spec.architecture === 'generator') duration *= 1.5

    // Ajustar por epochs
    const epochs = spec.parameters?.epochs || 10
    duration *= (epochs / 10)

    // Ajustar por batch size (batch más grande = más rápido)
    const batchSize = spec.parameters?.batchSize || 32
    duration *= (32 / batchSize)

    return Math.round(duration)
  }

  /**
   * Evaluar complejidad
   */
  private assessComplexity(spec: ModelSpec): 'low' | 'medium' | 'high' {
    let score = 0

    if (spec.architecture === 'transformer') score += 2
    if (spec.architecture === 'generator') score += 1
    if (spec.parameters?.epochs && spec.parameters.epochs > 50) score += 1
    if (spec.parameters?.batchSize && spec.parameters.batchSize > 128) score += 1

    if (score <= 1) return 'low'
    if (score <= 3) return 'medium'
    return 'high'
  }

  /**
   * Estimar recursos
   */
  private estimateResources(spec: ModelSpec): {
    gpu: boolean
    memory: string
    storage: string
  } {
    const complexity = this.assessComplexity(spec)
    const batchSize = spec.parameters?.batchSize || 32

    let gpu = false
    let memory = '4GB'
    let storage = '1GB'

    if (complexity === 'high' || spec.architecture === 'transformer') {
      gpu = true
      memory = '16GB'
      storage = '5GB'
    } else if (complexity === 'medium') {
      gpu = batchSize > 64
      memory = '8GB'
      storage = '2GB'
    }

    return { gpu, memory, storage }
  }

  /**
   * Limpiar historial
   */
  clearHistory(): void {
    this.contextHistory = []
  }
}

// Singleton instance
let enhancedAnalyzerInstance: EnhancedAdaptiveAnalyzer | null = null

export function getEnhancedAdaptiveAnalyzer(): EnhancedAdaptiveAnalyzer {
  if (!enhancedAnalyzerInstance) {
    enhancedAnalyzerInstance = new EnhancedAdaptiveAnalyzer()
  }
  return enhancedAnalyzerInstance
}

export default EnhancedAdaptiveAnalyzer










