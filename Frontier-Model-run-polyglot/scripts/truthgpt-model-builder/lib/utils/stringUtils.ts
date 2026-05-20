/**
 * Utilidades para Strings
 * ======================
 * 
 * Funciones útiles para trabajar con strings
 */

// ============================================================================
// TRANSFORMACIONES
// ============================================================================

/**
 * Convierte un string a slug (URL-friendly)
 */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/**
 * Convierte un string a camelCase
 */
export function toCamelCase(text: string): string {
  return text
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase()
    })
    .replace(/\s+/g, '')
}

/**
 * Convierte un string a PascalCase
 */
export function toPascalCase(text: string): string {
  return text
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word) => {
      return word.toUpperCase()
    })
    .replace(/\s+/g, '')
}

/**
 * Convierte un string a snake_case
 */
export function toSnakeCase(text: string): string {
  return text
    .replace(/([A-Z])/g, '_$1')
    .toLowerCase()
    .replace(/^_/, '')
    .replace(/\s+/g, '_')
}

/**
 * Convierte un string a kebab-case
 */
export function toKebabCase(text: string): string {
  return text
    .replace(/([A-Z])/g, '-$1')
    .toLowerCase()
    .replace(/^-/, '')
    .replace(/\s+/g, '-')
}

/**
 * Capitaliza la primera letra
 */
export function capitalize(text: string): string {
  if (!text) return text
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase()
}

/**
 * Capitaliza cada palabra
 */
export function capitalizeWords(text: string): string {
  return text
    .split(' ')
    .map(word => capitalize(word))
    .join(' ')
}

// ============================================================================
// VALIDACIÓN
// ============================================================================

/**
 * Verifica si un string es un email válido
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

/**
 * Verifica si un string es una URL válida
 */
export function isValidURL(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

/**
 * Verifica si un string es un número
 */
export function isNumeric(text: string): boolean {
  return !isNaN(Number(text)) && !isNaN(parseFloat(text))
}

/**
 * Verifica si un string está vacío o solo tiene espacios
 */
export function isEmpty(text: string): boolean {
  return !text || text.trim().length === 0
}

/**
 * Verifica si un string contiene solo letras
 */
export function isAlpha(text: string): boolean {
  return /^[a-zA-Z]+$/.test(text)
}

/**
 * Verifica si un string contiene solo números
 */
export function isNumericOnly(text: string): boolean {
  return /^\d+$/.test(text)
}

/**
 * Verifica si un string es alfanumérico
 */
export function isAlphanumeric(text: string): boolean {
  return /^[a-zA-Z0-9]+$/.test(text)
}

// ============================================================================
// MANIPULACIÓN
// ============================================================================

/**
 * Trunca un string con ellipsis
 */
export function truncate(text: string, maxLength: number, suffix: string = '...'): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength - suffix.length) + suffix
}

/**
 * Trunca un string en palabras completas
 */
export function truncateWords(text: string, maxWords: number, suffix: string = '...'): string {
  const words = text.split(' ')
  if (words.length <= maxWords) return text
  return words.slice(0, maxWords).join(' ') + suffix
}

/**
 * Elimina espacios duplicados
 */
export function removeExtraSpaces(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

/**
 * Elimina caracteres especiales
 */
export function removeSpecialChars(text: string, keep: string = ''): string {
  const regex = new RegExp(`[^a-zA-Z0-9${keep.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}]`, 'g')
  return text.replace(regex, '')
}

/**
 * Reemplaza múltiples ocurrencias
 */
export function replaceAll(text: string, search: string, replace: string): string {
  return text.split(search).join(replace)
}

/**
 * Invierte un string
 */
export function reverse(text: string): string {
  return text.split('').reverse().join('')
}

// ============================================================================
// EXTRACCIÓN
// ============================================================================

/**
 * Extrae números de un string
 */
export function extractNumbers(text: string): number[] {
  const matches = text.match(/\d+/g)
  return matches ? matches.map(Number) : []
}

/**
 * Extrae emails de un string
 */
export function extractEmails(text: string): string[] {
  const emailRegex = /[^\s@]+@[^\s@]+\.[^\s@]+/g
  return text.match(emailRegex) || []
}

/**
 * Extrae URLs de un string
 */
export function extractURLs(text: string): string[] {
  const urlRegex = /https?:\/\/[^\s]+/g
  return text.match(urlRegex) || []
}

/**
 * Extrae palabras de un string
 */
export function extractWords(text: string): string[] {
  return text.match(/\b\w+\b/g) || []
}

// ============================================================================
// FORMATO
// ============================================================================

/**
 * Formatea un string como título
 */
export function toTitle(text: string): string {
  return text
    .split(/[\s_-]+/)
    .map(word => capitalize(word))
    .join(' ')
}

/**
 * Agrega padding a un string
 */
export function pad(text: string, length: number, char: string = ' ', side: 'left' | 'right' | 'both' = 'right'): string {
  const padding = char.repeat(Math.max(0, length - text.length))
  
  switch (side) {
    case 'left':
      return padding + text
    case 'right':
      return text + padding
    case 'both':
      const half = Math.floor(padding.length / 2)
      return padding.slice(0, half) + text + padding.slice(half)
    default:
      return text
  }
}

/**
 * Formatea un string con máscara
 */
export function mask(text: string, pattern: string, placeholder: string = '_'): string {
  let result = ''
  let textIndex = 0

  for (let i = 0; i < pattern.length && textIndex < text.length; i++) {
    if (pattern[i] === placeholder) {
      result += text[textIndex]
      textIndex++
    } else {
      result += pattern[i]
    }
  }

  return result
}

/**
 * Oculta parte de un string (útil para emails, números de tarjeta, etc.)
 */
export function maskString(text: string, visibleStart: number = 0, visibleEnd: number = 0, maskChar: string = '*'): string {
  if (text.length <= visibleStart + visibleEnd) return text
  
  const start = text.slice(0, visibleStart)
  const end = text.slice(-visibleEnd)
  const middle = maskChar.repeat(text.length - visibleStart - visibleEnd)
  
  return start + middle + end
}

// ============================================================================
// COMPARACIÓN
// ============================================================================

/**
 * Compara dos strings ignorando mayúsculas/minúsculas
 */
export function equalsIgnoreCase(str1: string, str2: string): boolean {
  return str1.toLowerCase() === str2.toLowerCase()
}

/**
 * Verifica si un string contiene otro (case-insensitive)
 */
export function containsIgnoreCase(text: string, search: string): boolean {
  return text.toLowerCase().includes(search.toLowerCase())
}

/**
 * Verifica si un string comienza con otro (case-insensitive)
 */
export function startsWithIgnoreCase(text: string, prefix: string): boolean {
  return text.toLowerCase().startsWith(prefix.toLowerCase())
}

/**
 * Verifica si un string termina con otro (case-insensitive)
 */
export function endsWithIgnoreCase(text: string, suffix: string): boolean {
  return text.toLowerCase().endsWith(suffix.toLowerCase())
}

// ============================================================================
// UTILIDADES
// ============================================================================

/**
 * Genera un string aleatorio
 */
export function randomString(length: number, chars: string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'): string {
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

/**
 * Cuenta palabras en un string
 */
export function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(word => word.length > 0).length
}

/**
 * Cuenta caracteres sin espacios
 */
export function charCountNoSpaces(text: string): number {
  return text.replace(/\s/g, '').length
}

/**
 * Obtiene las primeras N palabras
 */
export function firstWords(text: string, count: number): string {
  return text.split(' ').slice(0, count).join(' ')
}

/**
 * Obtiene las últimas N palabras
 */
export function lastWords(text: string, count: number): string {
  const words = text.split(' ')
  return words.slice(-count).join(' ')
}







