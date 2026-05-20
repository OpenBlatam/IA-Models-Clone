/**
 * Utilidades de seguridad para modelos
 * =====================================
 */

/**
 * Sanitiza una descripción de modelo
 */
export function sanitizeDescription(description: string): string {
  // Eliminar caracteres peligrosos pero mantener el contenido útil
  return description
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '') // Remove scripts
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+\s*=/gi, '') // Remove event handlers
    .trim()
    .substring(0, 5000) // Limit length
}

/**
 * Valida que una descripción no contenga código malicioso
 */
export function validateDescriptionSecurity(description: string): {
  safe: boolean
  issues: string[]
} {
  const issues: string[] = []

  // Detectar scripts
  if (/<script/i.test(description)) {
    issues.push('Contiene etiquetas script')
  }

  // Detectar javascript: protocol
  if (/javascript:/i.test(description)) {
    issues.push('Contiene protocolo javascript')
  }

  // Detectar event handlers
  if (/on\w+\s*=/i.test(description)) {
    issues.push('Contiene event handlers')
  }

  // Detectar SQL injection patterns (básico)
  if (/(union|select|insert|update|delete|drop|exec|execute)/i.test(description)) {
    // Solo alertar si parece SQL, no bloquear completamente
    issues.push('Posible patrón SQL detectado')
  }

  // Detectar XSS patterns
  if (/<[^>]*>/g.test(description) && description.match(/<[^>]*>/g)!.length > 5) {
    issues.push('Demasiadas etiquetas HTML detectadas')
  }

  return {
    safe: issues.length === 0,
    issues
  }
}

/**
 * Sanitiza una especificación de modelo
 */
export function sanitizeModelSpec(spec: any): any {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
    return null
  }

  const sanitized: any = {}

  // Sanitizar campos permitidos
  const allowedFields = [
    'modelName', 'layers', 'optimizer', 'loss', 'metrics',
    'batch_size', 'epochs', 'learning_rate', 'validation_split'
  ]

  allowedFields.forEach(field => {
    if (spec[field] !== undefined) {
      sanitized[field] = spec[field]
    }
  })

  // Sanitizar layers
  if (Array.isArray(spec.layers)) {
    sanitized.layers = spec.layers.map((layer: any) => {
      if (!layer || typeof layer !== 'object') return null

      return {
        type: typeof layer.type === 'string' ? layer.type : undefined,
        params: typeof layer.params === 'object' && layer.params !== null
          ? layer.params
          : {}
      }
    }).filter((l: any) => l !== null)
  }

  // Sanitizar strings
  if (typeof sanitized.modelName === 'string') {
    sanitized.modelName = sanitizeDescription(sanitized.modelName)
  }

  return sanitized
}

/**
 * Valida el tamaño de una especificación
 */
export function validateSpecSize(spec: any): {
  valid: boolean
  size: number
  maxSize: number
  issues: string[]
} {
  const issues: string[] = []
  const maxSize = 100 * 1024 // 100KB
  const specString = JSON.stringify(spec)
  const size = new Blob([specString]).size

  if (size > maxSize) {
    issues.push(`Especificación demasiado grande: ${(size / 1024).toFixed(2)}KB (máximo: ${maxSize / 1024}KB)`)
  }

  // Validar profundidad de objetos
  const getDepth = (obj: any, depth = 0): number => {
    if (typeof obj !== 'object' || obj === null) return depth
    if (Array.isArray(obj)) {
      return Math.max(...obj.map(item => getDepth(item, depth + 1)))
    }
    return Math.max(...Object.values(obj).map(value => getDepth(value, depth + 1)))
  }

  const depth = getDepth(spec)
  if (depth > 10) {
    issues.push(`Profundidad de objeto excesiva: ${depth} niveles (máximo: 10)`)
  }

  return {
    valid: issues.length === 0,
    size,
    maxSize,
    issues
  }
}

/**
 * Crea una especificación segura desde una descripción
 */
export function createSafeSpec(description: string, spec?: any): any {
  const sanitizedDesc = sanitizeDescription(description)
  const securityCheck = validateDescriptionSecurity(sanitizedDesc)

  if (!securityCheck.safe) {
    throw new Error(`Descripción no segura: ${securityCheck.issues.join(', ')}`)
  }

  const sanitizedSpec = spec ? sanitizeModelSpec(spec) : null
  const sizeCheck = sanitizedSpec ? validateSpecSize(sanitizedSpec) : null

  if (sizeCheck && !sizeCheck.valid) {
    throw new Error(`Especificación inválida: ${sizeCheck.issues.join(', ')}`)
  }

  return {
    description: sanitizedDesc,
    spec: sanitizedSpec,
    securityCheck,
    sizeCheck
  }
}










