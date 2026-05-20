/**
 * Advanced Validator
 * Sistema de validaciones avanzadas para modelos
 */

import { ModelSpec } from './modules/management'

export interface ValidationRule {
  name: string
  validate: (spec: ModelSpec, description: string) => ValidationResult
  severity: 'error' | 'warning' | 'info'
}

export interface ValidationResult {
  isValid: boolean
  message: string
  severity: 'error' | 'warning' | 'info'
  suggestions?: string[]
}

export class AdvancedValidator {
  private rules: ValidationRule[] = []

  constructor() {
    this.registerDefaultRules()
  }

  /**
   * Registrar reglas por defecto
   */
  private registerDefaultRules(): void {
    // Validar descripción
    this.registerRule({
      name: 'description-length',
      severity: 'error',
      validate: (spec, description) => {
        if (description.length < 10) {
          return {
            isValid: false,
            message: 'La descripción debe tener al menos 10 caracteres',
            severity: 'error',
            suggestions: ['Proporciona más detalles sobre el modelo'],
          }
        }
        if (description.length > 5000) {
          return {
            isValid: false,
            message: 'La descripción es demasiado larga (máximo 5000 caracteres)',
            severity: 'error',
          }
        }
        return { isValid: true, message: '', severity: 'info' }
      },
    })

    // Validar arquitectura
    this.registerRule({
      name: 'architecture-compatibility',
      severity: 'warning',
      validate: (spec, description) => {
        const isVision = description.toLowerCase().includes('imagen') || 
                        description.toLowerCase().includes('vision') ||
                        description.toLowerCase().includes('cv')
        const isNLP = description.toLowerCase().includes('texto') ||
                     description.toLowerCase().includes('nlp') ||
                     description.toLowerCase().includes('lenguaje')

        if (isVision && spec.architecture !== 'cnn') {
          return {
            isValid: true,
            message: 'Para visión computacional, se recomienda usar arquitectura CNN',
            severity: 'warning',
            suggestions: ['Considera cambiar a arquitectura CNN'],
          }
        }

        if (isNLP && spec.architecture === 'cnn') {
          return {
            isValid: true,
            message: 'Para NLP, se recomienda usar arquitectura Transformer o LSTM',
            severity: 'warning',
            suggestions: ['Considera usar Transformer o LSTM'],
          }
        }

        return { isValid: true, message: '', severity: 'info' }
      },
    })

    // Validar learning rate
    this.registerRule({
      name: 'learning-rate-range',
      severity: 'warning',
      validate: (spec) => {
        if (spec.learningRate < 1e-6) {
          return {
            isValid: true,
            message: 'Learning rate muy bajo, puede requerir más épocas',
            severity: 'warning',
            suggestions: ['Considera aumentar el learning rate o el número de épocas'],
          }
        }
        if (spec.learningRate > 1e-2) {
          return {
            isValid: true,
            message: 'Learning rate muy alto, puede causar inestabilidad',
            severity: 'warning',
            suggestions: ['Considera reducir el learning rate'],
          }
        }
        return { isValid: true, message: '', severity: 'info' }
      },
    })

    // Validar batch size
    this.registerRule({
      name: 'batch-size',
      severity: 'info',
      validate: (spec) => {
        if (spec.batchSize > 128) {
          return {
            isValid: true,
            message: 'Batch size grande puede requerir más memoria',
            severity: 'info',
            suggestions: ['Asegúrate de tener suficiente memoria GPU'],
          }
        }
        return { isValid: true, message: '', severity: 'info' }
      },
    })

    // Validar épocas
    this.registerRule({
      name: 'epochs',
      severity: 'info',
      validate: (spec) => {
        if (spec.epochs < 1) {
          return {
            isValid: false,
            message: 'Debe haber al menos 1 época',
            severity: 'error',
          }
        }
        if (spec.epochs > 100) {
          return {
            isValid: true,
            message: 'Muchas épocas pueden requerir mucho tiempo',
            severity: 'info',
            suggestions: ['Considera usar early stopping'],
          }
        }
        return { isValid: true, message: '', severity: 'info' }
      },
    })
  }

  /**
   * Registrar una regla de validación
   */
  registerRule(rule: ValidationRule): void {
    this.rules.push(rule)
  }

  /**
   * Validar spec completo
   */
  validate(spec: ModelSpec, description: string): {
    isValid: boolean
    results: ValidationResult[]
    errors: ValidationResult[]
    warnings: ValidationResult[]
    info: ValidationResult[]
  } {
    const results: ValidationResult[] = []
    const errors: ValidationResult[] = []
    const warnings: ValidationResult[] = []
    const info: ValidationResult[] = []

    this.rules.forEach(rule => {
      const result = rule.validate(spec, description)
      results.push(result)

      if (!result.isValid || result.severity === 'error') {
        errors.push(result)
      } else if (result.severity === 'warning') {
        warnings.push(result)
      } else if (result.severity === 'info') {
        info.push(result)
      }
    })

    return {
      isValid: errors.length === 0,
      results,
      errors,
      warnings,
      info,
    }
  }

  /**
   * Validar solo errores críticos
   */
  validateCritical(spec: ModelSpec, description: string): boolean {
    const validation = this.validate(spec, description)
    return validation.errors.filter(e => !e.isValid).length === 0
  }

  /**
   * Obtener sugerencias de mejora
   */
  getSuggestions(spec: ModelSpec, description: string): string[] {
    const validation = this.validate(spec, description)
    const suggestions: string[] = []

    validation.results.forEach(result => {
      if (result.suggestions) {
        suggestions.push(...result.suggestions)
      }
    })

    return [...new Set(suggestions)] // Eliminar duplicados
  }
}

// Singleton instance
let validatorInstance: AdvancedValidator | null = null

export function getAdvancedValidator(): AdvancedValidator {
  if (!validatorInstance) {
    validatorInstance = new AdvancedValidator()
  }
  return validatorInstance
}

export default AdvancedValidator










