/**
 * Utilidades de Validación Avanzadas
 * ===================================
 * 
 * Funciones de validación complejas y composables
 */

/**
 * Resultado de validación
 */
export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings?: string[]
}

/**
 * Validador de campo
 */
export type FieldValidator<T = any> = (value: T, context?: any) => ValidationResult | string | boolean

/**
 * Valida un email
 */
export function validateEmail(email: string): ValidationResult {
  const errors: string[] = []
  
  if (!email) {
    errors.push('Email es requerido')
  } else {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      errors.push('Email inválido')
    }
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida una URL
 */
export function validateURL(url: string, options: {
  required?: boolean
  protocols?: string[]
} = {}): ValidationResult {
  const { required = false, protocols = ['http', 'https'] } = options
  const errors: string[] = []

  if (!url) {
    if (required) {
      errors.push('URL es requerida')
    }
    return { valid: !required, errors }
  }

  try {
    const urlObj = new URL(url)
    if (!protocols.includes(urlObj.protocol.replace(':', ''))) {
      errors.push(`Protocolo debe ser uno de: ${protocols.join(', ')}`)
    }
  } catch {
    errors.push('URL inválida')
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida un número
 */
export function validateNumber(
  value: any,
  options: {
    min?: number
    max?: number
    integer?: boolean
    required?: boolean
  } = {}
): ValidationResult {
  const { min, max, integer = false, required = false } = options
  const errors: string[] = []

  if (value === null || value === undefined || value === '') {
    if (required) {
      errors.push('Número es requerido')
    }
    return { valid: !required, errors }
  }

  const num = Number(value)

  if (isNaN(num)) {
    errors.push('Debe ser un número válido')
    return { valid: false, errors }
  }

  if (integer && !Number.isInteger(num)) {
    errors.push('Debe ser un número entero')
  }

  if (min !== undefined && num < min) {
    errors.push(`Debe ser mayor o igual a ${min}`)
  }

  if (max !== undefined && num > max) {
    errors.push(`Debe ser menor o igual a ${max}`)
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida una cadena
 */
export function validateString(
  value: any,
  options: {
    minLength?: number
    maxLength?: number
    pattern?: RegExp
    required?: boolean
    trim?: boolean
  } = {}
): ValidationResult {
  const { minLength, maxLength, pattern, required = false, trim = true } = options
  const errors: string[] = []

  const str = typeof value === 'string' ? (trim ? value.trim() : value) : String(value || '')

  if (!str) {
    if (required) {
      errors.push('Campo es requerido')
    }
    return { valid: !required, errors }
  }

  if (minLength !== undefined && str.length < minLength) {
    errors.push(`Debe tener al menos ${minLength} caracteres`)
  }

  if (maxLength !== undefined && str.length > maxLength) {
    errors.push(`Debe tener máximo ${maxLength} caracteres`)
  }

  if (pattern && !pattern.test(str)) {
    errors.push('Formato inválido')
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida una fecha
 */
export function validateDate(
  value: any,
  options: {
    min?: Date
    max?: Date
    required?: boolean
  } = {}
): ValidationResult {
  const { min, max, required = false } = options
  const errors: string[] = []

  if (!value) {
    if (required) {
      errors.push('Fecha es requerida')
    }
    return { valid: !required, errors }
  }

  const date = value instanceof Date ? value : new Date(value)

  if (isNaN(date.getTime())) {
    errors.push('Fecha inválida')
    return { valid: false, errors }
  }

  if (min && date < min) {
    errors.push(`Fecha debe ser posterior a ${min.toLocaleDateString()}`)
  }

  if (max && date > max) {
    errors.push(`Fecha debe ser anterior a ${max.toLocaleDateString()}`)
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida un array
 */
export function validateArray<T>(
  value: any,
  options: {
    minLength?: number
    maxLength?: number
    itemValidator?: (item: T) => ValidationResult
    required?: boolean
  } = {}
): ValidationResult {
  const { minLength, maxLength, itemValidator, required = false } = options
  const errors: string[] = []

  if (!Array.isArray(value)) {
    if (required) {
      errors.push('Debe ser un array')
    }
    return { valid: !required, errors }
  }

  if (minLength !== undefined && value.length < minLength) {
    errors.push(`Debe tener al menos ${minLength} elementos`)
  }

  if (maxLength !== undefined && value.length > maxLength) {
    errors.push(`Debe tener máximo ${maxLength} elementos`)
  }

  if (itemValidator) {
    value.forEach((item, index) => {
      const result = itemValidator(item)
      if (!result.valid) {
        errors.push(`Elemento ${index + 1}: ${result.errors.join(', ')}`)
      }
    })
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida un objeto
 */
export function validateObject<T extends Record<string, any>>(
  value: any,
  schema: {
    [K in keyof T]?: FieldValidator<T[K]> | FieldValidator<T[K]>[]
  },
  options: {
    required?: boolean
    strict?: boolean // Rechazar propiedades no definidas en el schema
  } = {}
): ValidationResult {
  const { required = false, strict = false } = options
  const errors: string[] = []

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    if (required) {
      errors.push('Debe ser un objeto')
    }
    return { valid: !required, errors }
  }

  // Validar propiedades del schema
  for (const [key, validators] of Object.entries(schema)) {
    const fieldValue = value[key]
    const validatorArray = Array.isArray(validators) ? validators : [validators]

    for (const validator of validatorArray) {
      const result = validator(fieldValue, value)
      
      if (typeof result === 'boolean') {
        if (!result) {
          errors.push(`Campo '${key}' es inválido`)
        }
      } else if (typeof result === 'string') {
        if (result) {
          errors.push(`Campo '${key}': ${result}`)
        }
      } else {
        if (!result.valid) {
          errors.push(`Campo '${key}': ${result.errors.join(', ')}`)
        }
      }
    }
  }

  // Validar propiedades no definidas (strict mode)
  if (strict) {
    const schemaKeys = Object.keys(schema)
    const valueKeys = Object.keys(value)
    const extraKeys = valueKeys.filter(key => !schemaKeys.includes(key))
    
    if (extraKeys.length > 0) {
      errors.push(`Propiedades no permitidas: ${extraKeys.join(', ')}`)
    }
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Crea un validador compuesto
 */
export function composeValidators<T>(
  ...validators: FieldValidator<T>[]
): FieldValidator<T> {
  return (value: T, context?: any) => {
    const errors: string[] = []

    for (const validator of validators) {
      const result = validator(value, context)
      
      if (typeof result === 'boolean') {
        if (!result) {
          errors.push('Validación falló')
        }
      } else if (typeof result === 'string') {
        if (result) {
          errors.push(result)
        }
      } else {
        errors.push(...result.errors)
      }
    }

    return {
      valid: errors.length === 0,
      errors
    }
  }
}

/**
 * Valida condicionalmente
 */
export function conditionalValidator<T>(
  condition: (value: T, context?: any) => boolean,
  validator: FieldValidator<T>
): FieldValidator<T> {
  return (value: T, context?: any) => {
    if (condition(value, context)) {
      return validator(value, context)
    }
    return { valid: true, errors: [] }
  }
}

/**
 * Valida formato de teléfono
 */
export function validatePhone(phone: string, required: boolean = false): ValidationResult {
  const errors: string[] = []

  if (!phone) {
    if (required) {
      errors.push('Teléfono es requerido')
    }
    return { valid: !required, errors }
  }

  // Formato básico: números, espacios, guiones, paréntesis, +
  const phoneRegex = /^[\d\s\-\(\)\+]+$/
  if (!phoneRegex.test(phone)) {
    errors.push('Formato de teléfono inválido')
  }

  // Al menos 10 dígitos
  const digits = phone.replace(/\D/g, '')
  if (digits.length < 10) {
    errors.push('Teléfono debe tener al menos 10 dígitos')
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida formato de código postal
 */
export function validatePostalCode(
  code: string,
  pattern?: RegExp,
  required: boolean = false
): ValidationResult {
  const errors: string[] = []

  if (!code) {
    if (required) {
      errors.push('Código postal es requerido')
    }
    return { valid: !required, errors }
  }

  if (pattern && !pattern.test(code)) {
    errors.push('Formato de código postal inválido')
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Valida contraseña
 */
export function validatePassword(
  password: string,
  options: {
    minLength?: number
    requireUppercase?: boolean
    requireLowercase?: boolean
    requireNumbers?: boolean
    requireSpecial?: boolean
  } = {}
): ValidationResult {
  const {
    minLength = 8,
    requireUppercase = false,
    requireLowercase = false,
    requireNumbers = false,
    requireSpecial = false
  } = options

  const errors: string[] = []
  const warnings: string[] = []

  if (!password) {
    errors.push('Contraseña es requerida')
    return { valid: false, errors }
  }

  if (password.length < minLength) {
    errors.push(`Contraseña debe tener al menos ${minLength} caracteres`)
  }

  if (requireUppercase && !/[A-Z]/.test(password)) {
    errors.push('Contraseña debe contener al menos una mayúscula')
  }

  if (requireLowercase && !/[a-z]/.test(password)) {
    errors.push('Contraseña debe contener al menos una minúscula')
  }

  if (requireNumbers && !/\d/.test(password)) {
    errors.push('Contraseña debe contener al menos un número')
  }

  if (requireSpecial && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    errors.push('Contraseña debe contener al menos un carácter especial')
  }

  // Warnings para seguridad adicional
  if (password.length < 12) {
    warnings.push('Se recomienda usar contraseñas de al menos 12 caracteres')
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  }
}






