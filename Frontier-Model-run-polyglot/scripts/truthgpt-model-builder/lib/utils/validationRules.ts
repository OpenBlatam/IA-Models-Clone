/**
 * Reglas de Validación Comunes
 * ============================
 * 
 * Reglas de validación reutilizables
 */

import { ValidationRule } from '../hooks/useForm'

// ============================================================================
// VALIDACIONES BÁSICAS
// ============================================================================

/**
 * Crea una regla de requerido
 */
export function required(message: string = 'Este campo es requerido'): ValidationRule {
  return {
    validator: (value: unknown) => {
      if (value === null || value === undefined) return false
      if (typeof value === 'string' && value.trim().length === 0) return false
      if (Array.isArray(value) && value.length === 0) return false
      return true
    },
    message
  }
}

/**
 * Crea una regla de longitud mínima
 */
export function minLength(min: number, message?: string): ValidationRule<string> {
  return {
    validator: (value: string) => value.length >= min,
    message: message || `Debe tener al menos ${min} caracteres`
  }
}

/**
 * Crea una regla de longitud máxima
 */
export function maxLength(max: number, message?: string): ValidationRule<string> {
  return {
    validator: (value: string) => value.length <= max,
    message: message || `Debe tener máximo ${max} caracteres`
  }
}

/**
 * Crea una regla de longitud exacta
 */
export function exactLength(length: number, message?: string): ValidationRule<string> {
  return {
    validator: (value: string) => value.length === length,
    message: message || `Debe tener exactamente ${length} caracteres`
  }
}

// ============================================================================
// VALIDACIONES DE EMAIL
// ============================================================================

/**
 * Valida email
 */
export function email(message: string = 'Email inválido'): ValidationRule<string> {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return {
    validator: (value: string) => emailRegex.test(value),
    message
  }
}

// ============================================================================
// VALIDACIONES DE URL
// ============================================================================

/**
 * Valida URL
 */
export function url(message: string = 'URL inválida'): ValidationRule<string> {
  return {
    validator: (value: string) => {
      try {
        new URL(value)
        return true
      } catch {
        return false
      }
    },
    message
  }
}

// ============================================================================
// VALIDACIONES NUMÉRICAS
// ============================================================================

/**
 * Valida número mínimo
 */
export function min(minValue: number, message?: string): ValidationRule<number> {
  return {
    validator: (value: number) => value >= minValue,
    message: message || `Debe ser mayor o igual a ${minValue}`
  }
}

/**
 * Valida número máximo
 */
export function max(maxValue: number, message?: string): ValidationRule<number> {
  return {
    validator: (value: number) => value <= maxValue,
    message: message || `Debe ser menor o igual a ${maxValue}`
  }
}

/**
 * Valida rango numérico
 */
export function range(minValue: number, maxValue: number, message?: string): ValidationRule<number> {
  return {
    validator: (value: number) => value >= minValue && value <= maxValue,
    message: message || `Debe estar entre ${minValue} y ${maxValue}`
  }
}

/**
 * Valida número entero
 */
export function integer(message: string = 'Debe ser un número entero'): ValidationRule<number> {
  return {
    validator: (value: number) => Number.isInteger(value),
    message
  }
}

/**
 * Valida número positivo
 */
export function positive(message: string = 'Debe ser un número positivo'): ValidationRule<number> {
  return {
    validator: (value: number) => value > 0,
    message
  }
}

// ============================================================================
// VALIDACIONES DE PATRÓN
// ============================================================================

/**
 * Valida patrón regex
 */
export function pattern(regex: RegExp, message: string): ValidationRule<string> {
  return {
    validator: (value: string) => regex.test(value),
    message
  }
}

/**
 * Valida alfanumérico
 */
export function alphanumeric(message: string = 'Solo se permiten letras y números'): ValidationRule<string> {
  return {
    validator: (value: string) => /^[a-zA-Z0-9]+$/.test(value),
    message
  }
}

/**
 * Valida solo letras
 */
export function alpha(message: string = 'Solo se permiten letras'): ValidationRule<string> {
  return {
    validator: (value: string) => /^[a-zA-Z]+$/.test(value),
    message
  }
}

/**
 * Valida solo números
 */
export function numeric(message: string = 'Solo se permiten números'): ValidationRule<string> {
  return {
    validator: (value: string) => /^\d+$/.test(value),
    message
  }
}

// ============================================================================
// VALIDACIONES PERSONALIZADAS
// ============================================================================

/**
 * Crea una regla de validación personalizada
 */
export function custom<T>(
  validator: (value: T) => boolean,
  message: string
): ValidationRule<T> {
  return { validator, message }
}

/**
 * Valida que dos valores sean iguales
 */
export function match(
  otherValue: unknown,
  message: string = 'Los valores no coinciden'
): ValidationRule<unknown> {
  return {
    validator: (value: unknown) => value === otherValue,
    message
  }
}

/**
 * Valida que un array tenga mínimo elementos
 */
export function minItems(min: number, message?: string): ValidationRule<unknown[]> {
  return {
    validator: (value: unknown[]) => value.length >= min,
    message: message || `Debe tener al menos ${min} elementos`
  }
}

/**
 * Valida que un array tenga máximo elementos
 */
export function maxItems(max: number, message?: string): ValidationRule<unknown[]> {
  return {
    validator: (value: unknown[]) => value.length <= max,
    message: message || `Debe tener máximo ${max} elementos`
  }
}

// ============================================================================
// COMBINADORES
// ============================================================================

/**
 * Combina múltiples reglas con AND
 */
export function and<T>(...rules: ValidationRule<T>[]): ValidationRule<T> {
  return {
    validator: (value: T) => rules.every(rule => rule.validator(value)),
    message: rules.map(r => r.message).join(', ')
  }
}

/**
 * Combina múltiples reglas con OR
 */
export function or<T>(...rules: ValidationRule<T>[]): ValidationRule<T> {
  return {
    validator: (value: T) => rules.some(rule => rule.validator(value)),
    message: rules.map(r => r.message).join(' o ')
  }
}







