/**
 * Utility functions for validation
 */

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings?: string[]
}

/**
 * Validate message content
 */
export function validateMessageContent(content: string): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  if (!content || content.trim().length === 0) {
    errors.push('El contenido del mensaje no puede estar vacío')
  }

  if (content.length > 10000) {
    errors.push('El mensaje es demasiado largo (máximo 10,000 caracteres)')
  }

  if (content.length > 5000) {
    warnings.push('El mensaje es muy largo (más de 5,000 caracteres)')
  }

  // Check for potentially dangerous content
  if (content.includes('<script')) {
    errors.push('El mensaje contiene código potencialmente peligroso')
  }

  // Check for spam patterns
  const spamPatterns = [
    /(.)\1{10,}/, // Repeated characters
    /(http[s]?:\/\/){3,}/, // Multiple URLs
  ]

  for (const pattern of spamPatterns) {
    if (pattern.test(content)) {
      warnings.push('El mensaje puede ser spam')
      break
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings: warnings.length > 0 ? warnings : undefined,
  }
}

/**
 * Validate search query
 */
export function validateSearchQuery(query: string): ValidationResult {
  const errors: string[] = []

  if (query.length > 200) {
    errors.push('La búsqueda es demasiado larga (máximo 200 caracteres)')
  }

  // Check for dangerous regex patterns
  if (query.includes('.*') && query.length > 50) {
    errors.push('La búsqueda contiene patrones complejos que pueden ser lentos')
  }

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * Validate export format
 */
export function validateExportFormat(format: string): ValidationResult {
  const validFormats = ['json', 'txt', 'md', 'html', 'csv', 'xml', 'yaml', 'pdf']
  const errors: string[] = []

  if (!validFormats.includes(format.toLowerCase())) {
    errors.push(`Formato no válido. Formatos permitidos: ${validFormats.join(', ')}`)
  }

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * Validate theme configuration
 */
export function validateThemeConfig(config: any): ValidationResult {
  const errors: string[] = []
  const requiredFields = ['background', 'foreground', 'primary']

  if (!config || typeof config !== 'object') {
    errors.push('La configuración del tema debe ser un objeto')
    return { valid: false, errors }
  }

  for (const field of requiredFields) {
    if (!(field in config)) {
      errors.push(`El campo "${field}" es requerido en la configuración del tema`)
    } else if (typeof config[field] !== 'string') {
      errors.push(`El campo "${field}" debe ser una cadena de texto`)
    } else if (!/^#[0-9A-Fa-f]{6}$/.test(config[field])) {
      errors.push(`El campo "${field}" debe ser un color hexadecimal válido`)
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * Validate notification settings
 */
export function validateNotificationSettings(settings: any): ValidationResult {
  const errors: string[] = []

  if (settings.sound !== undefined && typeof settings.sound !== 'boolean') {
    errors.push('El campo "sound" debe ser un booleano')
  }

  if (settings.desktop !== undefined && typeof settings.desktop !== 'boolean') {
    errors.push('El campo "desktop" debe ser un booleano')
  }

  if (settings.badge !== undefined && typeof settings.badge !== 'boolean') {
    errors.push('El campo "badge" debe ser un booleano')
  }

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * Validate collaboration data
 */
export function validateCollaborationData(data: any): ValidationResult {
  const errors: string[] = []

  if (!data || typeof data !== 'object') {
    errors.push('Los datos de colaboración deben ser un objeto')
    return { valid: false, errors }
  }

  if (data.collaborators && !Array.isArray(data.collaborators)) {
    errors.push('Los colaboradores deben ser un array')
  }

  if (data.messageSharing && !Array.isArray(data.messageSharing)) {
    errors.push('El compartir de mensajes debe ser un array')
  }

  return {
    valid: errors.length === 0,
    errors,
  }
}

/**
 * Sanitize user input
 */
export function sanitizeInput(input: string): string {
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+\s*=/gi, '')
    .trim()
}

/**
 * Validate file for import
 */
export function validateImportFile(file: File): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []
  const maxSize = 10 * 1024 * 1024 // 10MB
  const allowedTypes = ['application/json', 'text/plain', 'text/markdown', 'text/csv']

  if (file.size > maxSize) {
    errors.push(`El archivo es demasiado grande (máximo ${maxSize / 1024 / 1024}MB)`)
  }

  if (file.size > maxSize / 2) {
    warnings.push('El archivo es grande, la importación puede tardar')
  }

  if (!allowedTypes.includes(file.type) && !file.name.match(/\.(json|txt|md|csv)$/i)) {
    errors.push('Tipo de archivo no soportado. Use JSON, TXT, MD o CSV')
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings: warnings.length > 0 ? warnings : undefined,
  }
}




