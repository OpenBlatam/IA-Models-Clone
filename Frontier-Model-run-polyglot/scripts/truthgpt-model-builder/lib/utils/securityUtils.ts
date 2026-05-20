/**
 * Utilidades de Seguridad
 * =======================
 * 
 * Funciones para sanitización y validación de seguridad
 */

/**
 * Sanitiza una cadena HTML removiendo etiquetas peligrosas
 */
export function sanitizeHTML(html: string): string {
  if (typeof window === 'undefined') return html

  const div = document.createElement('div')
  div.textContent = html
  return div.innerHTML
}

/**
 * Escapa caracteres HTML especiales
 */
export function escapeHTML(str: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  }
  return str.replace(/[&<>"']/g, (m) => map[m])
}

/**
 * Desescapa caracteres HTML
 */
export function unescapeHTML(str: string): string {
  const map: Record<string, string> = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#039;': "'"
  }
  return str.replace(/&amp;|&lt;|&gt;|&quot;|&#039;/g, (m) => map[m])
}

/**
 * Valida si una cadena contiene código malicioso
 */
export function containsMaliciousCode(str: string): boolean {
  const dangerousPatterns = [
    /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi,
    /<iframe/gi,
    /<object/gi,
    /<embed/gi,
    /eval\s*\(/gi,
    /expression\s*\(/gi
  ]

  return dangerousPatterns.some(pattern => pattern.test(str))
}

/**
 * Sanitiza una URL removiendo javascript: y otros protocolos peligrosos
 */
export function sanitizeURL(url: string): string | null {
  try {
    const urlObj = new URL(url, window.location.origin)
    
    // Solo permitir http, https, mailto, tel
    const allowedProtocols = ['http:', 'https:', 'mailto:', 'tel:']
    if (!allowedProtocols.includes(urlObj.protocol)) {
      return null
    }

    return urlObj.toString()
  } catch {
    return null
  }
}

/**
 * Valida si una cadena es un email seguro
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email) && !containsMaliciousCode(email)
}

/**
 * Valida si una cadena es una URL segura
 */
export function isValidURL(url: string): boolean {
  try {
    const urlObj = new URL(url)
    return ['http:', 'https:'].includes(urlObj.protocol)
  } catch {
    return false
  }
}

/**
 * Genera un token aleatorio seguro
 */
export function generateSecureToken(length: number = 32): string {
  const array = new Uint8Array(length)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(array)
  } else {
    // Fallback para entornos sin crypto
    for (let i = 0; i < length; i++) {
      array[i] = Math.floor(Math.random() * 256)
    }
  }
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('')
}

/**
 * Hashea una cadena (SHA-256)
 */
export async function hashString(str: string): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('Web Crypto API not available')
  }

  const encoder = new TextEncoder()
  const data = encoder.encode(str)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * Valida y sanitiza entrada de usuario
 */
export function sanitizeUserInput(input: string, options: {
  maxLength?: number
  allowHTML?: boolean
  trim?: boolean
} = {}): string {
  const {
    maxLength = 10000,
    allowHTML = false,
    trim = true
  } = options

  let sanitized = input

  if (trim) {
    sanitized = sanitized.trim()
  }

  if (sanitized.length > maxLength) {
    sanitized = sanitized.substring(0, maxLength)
  }

  if (!allowHTML) {
    sanitized = escapeHTML(sanitized)
  } else {
    sanitized = sanitizeHTML(sanitized)
  }

  return sanitized
}

/**
 * Valida si una cadena es segura para usar en SQL (prevención básica de SQL injection)
 */
export function isSQLSafe(str: string): boolean {
  const dangerousSQL = [
    /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)/gi,
    /(--|#|\/\*|\*\/)/g,
    /(;|\||&)/g
  ]

  return !dangerousSQL.some(pattern => pattern.test(str))
}

/**
 * Crea una función de validación de entrada segura
 */
export function createSafeValidator(options: {
  maxLength?: number
  minLength?: number
  pattern?: RegExp
  required?: boolean
  sanitize?: boolean
}) {
  return (input: string): { valid: boolean; error?: string; sanitized?: string } => {
    const {
      maxLength = 10000,
      minLength = 0,
      pattern,
      required = false,
      sanitize = true
    } = options

    if (required && !input.trim()) {
      return { valid: false, error: 'Campo requerido' }
    }

    if (input.length < minLength) {
      return { valid: false, error: `Mínimo ${minLength} caracteres` }
    }

    if (input.length > maxLength) {
      return { valid: false, error: `Máximo ${maxLength} caracteres` }
    }

    if (pattern && !pattern.test(input)) {
      return { valid: false, error: 'Formato inválido' }
    }

    if (containsMaliciousCode(input)) {
      return { valid: false, error: 'Contenido no permitido' }
    }

    const sanitized = sanitize ? sanitizeUserInput(input) : input

    return { valid: true, sanitized }
  }
}






