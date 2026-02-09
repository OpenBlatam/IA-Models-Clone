/**
 * String utility functions
 */

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number, suffix: string = '...'): string {
  if (str.length <= maxLength) return str
  return str.substring(0, maxLength - suffix.length) + suffix
}

/**
 * Truncate words
 */
export function truncateWords(str: string, maxWords: number, suffix: string = '...'): string {
  const words = str.split(/\s+/)
  if (words.length <= maxWords) return str
  return words.slice(0, maxWords).join(' ') + suffix
}

/**
 * Capitalize first letter
 */
export function capitalize(str: string): string {
  if (!str) return str
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase()
}

/**
 * Capitalize each word
 */
export function capitalizeWords(str: string): string {
  return str.replace(/\b\w/g, char => char.toUpperCase())
}

/**
 * Remove HTML tags
 */
export function stripHTML(html: string): string {
  const tmp = document.createElement('DIV')
  tmp.innerHTML = html
  return tmp.textContent || tmp.innerText || ''
}

/**
 * Escape HTML
 */
export function escapeHTML(str: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  }
  return str.replace(/[&<>"']/g, m => map[m])
}

/**
 * Unescape HTML
 */
export function unescapeHTML(str: string): string {
  const map: Record<string, string> = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#039;': "'",
  }
  return str.replace(/&amp;|&lt;|&gt;|&quot;|&#039;/g, m => map[m])
}

/**
 * Slugify string
 */
export function slugify(str: string): string {
  return str
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/**
 * Camel case
 */
export function camelCase(str: string): string {
  return str
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase()
    })
    .replace(/\s+/g, '')
}

/**
 * Kebab case
 */
export function kebabCase(str: string): string {
  return str
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]+/g, '-')
    .toLowerCase()
}

/**
 * Snake case
 */
export function snakeCase(str: string): string {
  return str
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
}

/**
 * Remove accents
 */
export function removeAccents(str: string): string {
  return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
}

/**
 * Count words
 */
export function countWords(str: string): number {
  return str.trim().split(/\s+/).filter(word => word.length > 0).length
}

/**
 * Count characters (excluding spaces)
 */
export function countCharacters(str: string, excludeSpaces: boolean = false): number {
  if (excludeSpaces) {
    return str.replace(/\s/g, '').length
  }
  return str.length
}

/**
 * Extract URLs
 */
export function extractURLs(str: string): string[] {
  const urlRegex = /https?:\/\/[^\s]+/g
  return str.match(urlRegex) || []
}

/**
 * Extract emails
 */
export function extractEmails(str: string): string[] {
  const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g
  return str.match(emailRegex) || []
}

/**
 * Extract mentions (@username)
 */
export function extractMentions(str: string): string[] {
  const mentionRegex = /@(\w+)/g
  const matches = str.matchAll(mentionRegex)
  return Array.from(matches, m => m[1])
}

/**
 * Extract hashtags (#tag)
 */
export function extractHashtags(str: string): string[] {
  const hashtagRegex = /#(\w+)/g
  const matches = str.matchAll(hashtagRegex)
  return Array.from(matches, m => m[1])
}

/**
 * Highlight text
 */
export function highlightText(text: string, query: string, className: string = 'highlight'): string {
  if (!query) return text
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  return text.replace(regex, `<span class="${className}">$1</span>`)
}

/**
 * Mask sensitive data
 */
export function maskSensitive(str: string, visibleChars: number = 4): string {
  if (str.length <= visibleChars * 2) {
    return '*'.repeat(str.length)
  }
  const start = str.substring(0, visibleChars)
  const end = str.substring(str.length - visibleChars)
  const middle = '*'.repeat(str.length - visibleChars * 2)
  return `${start}${middle}${end}`
}

/**
 * Generate random string
 */
export function randomString(length: number = 10): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

/**
 * Check if string is empty or whitespace
 */
export function isEmpty(str: string | null | undefined): boolean {
  return !str || str.trim().length === 0
}

/**
 * Pad string
 */
export function pad(str: string, length: number, padChar: string = ' '): string {
  if (str.length >= length) return str
  const padding = padChar.repeat(length - str.length)
  return str + padding
}

/**
 * Pad left
 */
export function padLeft(str: string, length: number, padChar: string = ' '): string {
  if (str.length >= length) return str
  const padding = padChar.repeat(length - str.length)
  return padding + str
}




