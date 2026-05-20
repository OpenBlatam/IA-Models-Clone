/**
 * Utilidades de Validación de Esquemas
 * ====================================
 * 
 * Funciones para validar datos contra esquemas
 */

export type SchemaType = 'string' | 'number' | 'boolean' | 'object' | 'array' | 'null' | 'any'

export interface SchemaField {
  type: SchemaType | SchemaType[]
  required?: boolean
  default?: any
  validator?: (value: any) => boolean | string
  min?: number
  max?: number
  minLength?: number
  maxLength?: number
  pattern?: RegExp
  enum?: any[]
  items?: SchemaField // Para arrays
  properties?: Record<string, SchemaField> // Para objects
  additionalProperties?: boolean
}

export interface Schema {
  type: 'object'
  properties: Record<string, SchemaField>
  required?: string[]
  additionalProperties?: boolean
}

export interface ValidationResult {
  valid: boolean
  errors: Array<{
    path: string
    message: string
  }>
}

/**
 * Valida un valor contra un campo de esquema
 */
function validateField(
  value: any,
  field: SchemaField,
  path: string = ''
): Array<{ path: string; message: string }> {
  const errors: Array<{ path: string; message: string }> = []

  // Check required
  if (field.required && (value === undefined || value === null)) {
    errors.push({ path, message: 'Campo requerido' })
    return errors
  }

  // Skip validation if value is undefined/null and not required
  if (value === undefined || value === null) {
    return errors
  }

  // Check type
  const types = Array.isArray(field.type) ? field.type : [field.type]
  const valueType = Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value

  if (!types.includes(valueType as SchemaType) && !types.includes('any')) {
    errors.push({
      path,
      message: `Tipo esperado: ${types.join(' o ')}, recibido: ${valueType}`
    })
    return errors
  }

  // Type-specific validations
  if (valueType === 'string') {
    const str = value as string

    if (field.minLength !== undefined && str.length < field.minLength) {
      errors.push({
        path,
        message: `Longitud mínima: ${field.minLength}, recibido: ${str.length}`
      })
    }

    if (field.maxLength !== undefined && str.length > field.maxLength) {
      errors.push({
        path,
        message: `Longitud máxima: ${field.maxLength}, recibido: ${str.length}`
      })
    }

    if (field.pattern && !field.pattern.test(str)) {
      errors.push({ path, message: 'Formato inválido' })
    }

    if (field.enum && !field.enum.includes(str)) {
      errors.push({
        path,
        message: `Valor debe ser uno de: ${field.enum.join(', ')}`
      })
    }
  }

  if (valueType === 'number') {
    const num = value as number

    if (field.min !== undefined && num < field.min) {
      errors.push({
        path,
        message: `Valor mínimo: ${field.min}, recibido: ${num}`
      })
    }

    if (field.max !== undefined && num > field.max) {
      errors.push({
        path,
        message: `Valor máximo: ${field.max}, recibido: ${num}`
      })
    }

    if (field.enum && !field.enum.includes(num)) {
      errors.push({
        path,
        message: `Valor debe ser uno de: ${field.enum.join(', ')}`
      })
    }
  }

  if (valueType === 'array') {
    const arr = value as any[]

    if (field.minLength !== undefined && arr.length < field.minLength) {
      errors.push({
        path,
        message: `Mínimo de elementos: ${field.minLength}, recibido: ${arr.length}`
      })
    }

    if (field.maxLength !== undefined && arr.length > field.maxLength) {
      errors.push({
        path,
        message: `Máximo de elementos: ${field.maxLength}, recibido: ${arr.length}`
      })
    }

    if (field.items) {
      arr.forEach((item, index) => {
        const itemErrors = validateField(item, field.items!, `${path}[${index}]`)
        errors.push(...itemErrors)
      })
    }
  }

  if (valueType === 'object') {
    const obj = value as Record<string, any>

    if (field.properties) {
      // Validate defined properties
      for (const [key, propSchema] of Object.entries(field.properties)) {
        const propValue = obj[key]
        const propField: SchemaField = {
          ...propSchema,
          required: propSchema.required || field.required
        }
        const propErrors = validateField(propValue, propField, path ? `${path}.${key}` : key)
        errors.push(...propErrors)
      }

      // Check additional properties
      if (field.additionalProperties === false) {
        const allowedKeys = new Set(Object.keys(field.properties))
        for (const key of Object.keys(obj)) {
          if (!allowedKeys.has(key)) {
            errors.push({
              path: path ? `${path}.${key}` : key,
              message: 'Propiedad no permitida'
            })
          }
        }
      }
    }
  }

  // Custom validator
  if (field.validator) {
    const result = field.validator(value)
    if (result !== true) {
      errors.push({
        path,
        message: typeof result === 'string' ? result : 'Validación falló'
      })
    }
  }

  return errors
}

/**
 * Valida un objeto contra un esquema
 */
export function validateSchema(data: any, schema: Schema): ValidationResult {
  const errors: Array<{ path: string; message: string }> = []

  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    return {
      valid: false,
      errors: [{ path: '', message: 'Debe ser un objeto' }]
    }
  }

  // Validate required fields
  if (schema.required) {
    for (const key of schema.required) {
      if (!(key in data) || data[key] === undefined || data[key] === null) {
        errors.push({ path: key, message: 'Campo requerido' })
      }
    }
  }

  // Validate properties
  for (const [key, field] of Object.entries(schema.properties)) {
    const value = data[key]
    const fieldWithRequired: SchemaField = {
      ...field,
      required: schema.required?.includes(key) || field.required
    }
    const fieldErrors = validateField(value, fieldWithRequired, key)
    errors.push(...fieldErrors)
  }

  // Check additional properties
  if (schema.additionalProperties === false) {
    const allowedKeys = new Set(Object.keys(schema.properties))
    for (const key of Object.keys(data)) {
      if (!allowedKeys.has(key)) {
        errors.push({ path: key, message: 'Propiedad no permitida' })
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

/**
 * Aplica valores por defecto desde un esquema
 */
export function applyDefaults(data: any, schema: Schema): any {
  const result = { ...data }

  for (const [key, field] of Object.entries(schema.properties)) {
    if (!(key in result) && field.default !== undefined) {
      result[key] = field.default
    } else if (field.type === 'object' && field.properties) {
      if (result[key] && typeof result[key] === 'object') {
        result[key] = applyDefaults(result[key], {
          type: 'object',
          properties: field.properties
        })
      }
    }
  }

  return result
}

/**
 * Crea un validador de esquema
 */
export function createSchemaValidator<T extends Record<string, any>>(schema: Schema) {
  return (data: any): { valid: boolean; data?: T; errors?: Array<{ path: string; message: string }> } => {
    const result = validateSchema(data, schema)
    
    if (result.valid) {
      const withDefaults = applyDefaults(data, schema)
      return { valid: true, data: withDefaults as T }
    }

    return { valid: false, errors: result.errors }
  }
}






