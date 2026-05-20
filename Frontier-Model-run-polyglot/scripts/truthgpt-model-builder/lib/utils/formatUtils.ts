/**
 * Utilidades de Formateo Mejoradas
 * ================================
 * 
 * Funciones para formatear datos de manera consistente
 */

/**
 * Formatea un número como moneda
 */
export function formatCurrency(
  amount: number,
  options: {
    currency?: string
    locale?: string
    minimumFractionDigits?: number
    maximumFractionDigits?: number
  } = {}
): string {
  const {
    currency = 'USD',
    locale = 'en-US',
    minimumFractionDigits = 2,
    maximumFractionDigits = 2
  } = options

  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    minimumFractionDigits,
    maximumFractionDigits
  }).format(amount)
}

/**
 * Formatea un número con separadores de miles
 */
export function formatNumber(
  number: number,
  options: {
    locale?: string
    minimumFractionDigits?: number
    maximumFractionDigits?: number
    notation?: 'standard' | 'scientific' | 'engineering' | 'compact'
  } = {}
): string {
  const {
    locale = 'en-US',
    minimumFractionDigits = 0,
    maximumFractionDigits = 3,
    notation = 'standard'
  } = options

  return new Intl.NumberFormat(locale, {
    minimumFractionDigits,
    maximumFractionDigits,
    notation
  }).format(number)
}

/**
 * Formatea un número como porcentaje
 */
export function formatPercent(
  value: number,
  options: {
    locale?: string
    minimumFractionDigits?: number
    maximumFractionDigits?: number
  } = {}
): string {
  const {
    locale = 'en-US',
    minimumFractionDigits = 0,
    maximumFractionDigits = 2
  } = options

  return new Intl.NumberFormat(locale, {
    style: 'percent',
    minimumFractionDigits,
    maximumFractionDigits
  }).format(value / 100)
}

/**
 * Formatea una fecha
 */
export function formatDate(
  date: Date | number | string,
  options: {
    locale?: string
    dateStyle?: 'full' | 'long' | 'medium' | 'short'
    timeStyle?: 'full' | 'long' | 'medium' | 'short'
    format?: string // Formato personalizado
  } = {}
): string {
  const { locale = 'en-US', dateStyle, timeStyle, format } = options

  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date

  if (isNaN(dateObj.getTime())) {
    return 'Fecha inválida'
  }

  if (format) {
    // Formato personalizado simple
    return formatCustomDate(dateObj, format)
  }

  const formatOptions: Intl.DateTimeFormatOptions = {}
  if (dateStyle) formatOptions.dateStyle = dateStyle
  if (timeStyle) formatOptions.timeStyle = timeStyle

  return new Intl.DateTimeFormat(locale, formatOptions).format(dateObj)
}

/**
 * Formatea una fecha con formato personalizado
 */
function formatCustomDate(date: Date, format: string): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  const seconds = String(date.getSeconds()).padStart(2, '0')

  return format
    .replace('YYYY', String(year))
    .replace('MM', month)
    .replace('DD', day)
    .replace('HH', hours)
    .replace('mm', minutes)
    .replace('ss', seconds)
}

/**
 * Formatea un tamaño de archivo
 */
export function formatFileSize(
  bytes: number,
  options: {
    decimals?: number
    binary?: boolean
  } = {}
): string {
  const { decimals = 2, binary = false } = options

  if (bytes === 0) return '0 Bytes'

  const k = binary ? 1024 : 1000
  const dm = decimals < 0 ? 0 : decimals
  const sizes = binary
    ? ['Bytes', 'KiB', 'MiB', 'GiB', 'TiB']
    : ['Bytes', 'KB', 'MB', 'GB', 'TB']

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/**
 * Formatea un número de teléfono
 */
export function formatPhoneNumber(
  phone: string,
  format: 'US' | 'INTERNATIONAL' | 'E164' = 'US'
): string {
  const digits = phone.replace(/\D/g, '')

  if (format === 'E164') {
    return `+${digits}`
  }

  if (format === 'INTERNATIONAL') {
    if (digits.length === 10) {
      return `+1 ${digits.slice(0, 3)} ${digits.slice(3, 6)} ${digits.slice(6)}`
    }
    return `+${digits}`
  }

  // US format: (123) 456-7890
  if (digits.length === 10) {
    return `(${digits.slice(0, 3)}) ${digits.slice(3, 6)}-${digits.slice(6)}`
  }

  return phone
}

/**
 * Formatea un código postal
 */
export function formatPostalCode(
  code: string,
  format: 'US' | 'CA' | 'UK' = 'US'
): string {
  const cleaned = code.replace(/\s+/g, '').toUpperCase()

  if (format === 'US' && cleaned.length === 5) {
    return cleaned
  }

  if (format === 'US' && cleaned.length === 9) {
    return `${cleaned.slice(0, 5)}-${cleaned.slice(5)}`
  }

  if (format === 'CA' && cleaned.length === 6) {
    return `${cleaned.slice(0, 3)} ${cleaned.slice(3)}`
  }

  if (format === 'UK' && cleaned.length >= 5) {
    // Formato UK: SW1A 1AA
    if (cleaned.length === 6 || cleaned.length === 7) {
      return `${cleaned.slice(0, -3)} ${cleaned.slice(-3)}`
    }
  }

  return cleaned
}

/**
 * Formatea texto con truncamiento
 */
export function truncateText(
  text: string,
  maxLength: number,
  suffix: string = '...'
): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength - suffix.length) + suffix
}

/**
 * Formatea texto con ellipsis inteligente
 */
export function truncateTextSmart(
  text: string,
  maxLength: number,
  options: {
    preserveWords?: boolean
    suffix?: string
  } = {}
): string {
  const { preserveWords = true, suffix = '...' } = options

  if (text.length <= maxLength) return text

  if (!preserveWords) {
    return truncateText(text, maxLength, suffix)
  }

  // Truncar en el último espacio antes del límite
  const truncated = text.slice(0, maxLength - suffix.length)
  const lastSpace = truncated.lastIndexOf(' ')

  if (lastSpace > 0) {
    return truncated.slice(0, lastSpace) + suffix
  }

  return truncated + suffix
}

/**
 * Formatea texto con capitalización
 */
export function capitalize(text: string): string {
  if (!text) return text
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase()
}

/**
 * Formatea texto con título (cada palabra capitalizada)
 */
export function toTitleCase(text: string): string {
  return text.replace(/\w\S*/g, (txt) => {
    return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase()
  })
}

/**
 * Formatea texto con camelCase
 */
export function toCamelCase(text: string): string {
  return text
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase()
    })
    .replace(/\s+/g, '')
}

/**
 * Formatea texto con kebab-case
 */
export function toKebabCase(text: string): string {
  return text
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]+/g, '-')
    .toLowerCase()
}

/**
 * Formatea texto con snake_case
 */
export function toSnakeCase(text: string): string {
  return text
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
}

/**
 * Formatea un slug (URL-friendly)
 */
export function toSlug(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/**
 * Formatea un número con unidades (1K, 1M, etc.)
 */
export function formatCompactNumber(
  number: number,
  options: {
    locale?: string
    notation?: 'standard' | 'scientific' | 'engineering' | 'compact'
    compactDisplay?: 'short' | 'long'
  } = {}
): string {
  const {
    locale = 'en-US',
    notation = 'compact',
    compactDisplay = 'short'
  } = options

  return new Intl.NumberFormat(locale, {
    notation,
    compactDisplay
  }).format(number)
}

/**
 * Formatea un rango de números
 */
export function formatRange(
  start: number,
  end: number,
  separator: string = ' - '
): string {
  return `${start}${separator}${end}`
}

/**
 * Formatea una lista de elementos
 */
export function formatList(
  items: string[],
  options: {
    conjunction?: string
    separator?: string
  } = {}
): string {
  const { conjunction = 'y', separator = ', ' } = options

  if (items.length === 0) return ''
  if (items.length === 1) return items[0]
  if (items.length === 2) return `${items[0]} ${conjunction} ${items[1]}`

  const last = items[items.length - 1]
  const rest = items.slice(0, -1).join(separator)
  return `${rest}${separator}${conjunction} ${last}`
}
