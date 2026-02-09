/**
 * Date and time utility functions
 */

/**
 * Format date to relative time
 */
export function formatRelativeTime(date: Date | number): string {
  const now = Date.now()
  const timestamp = date instanceof Date ? date.getTime() : date
  const diff = now - timestamp
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  const weeks = Math.floor(days / 7)
  const months = Math.floor(days / 30)
  const years = Math.floor(days / 365)

  if (seconds < 60) return 'hace un momento'
  if (minutes < 60) return `hace ${minutes} ${minutes === 1 ? 'minuto' : 'minutos'}`
  if (hours < 24) return `hace ${hours} ${hours === 1 ? 'hora' : 'horas'}`
  if (days < 7) return `hace ${days} ${days === 1 ? 'día' : 'días'}`
  if (weeks < 4) return `hace ${weeks} ${weeks === 1 ? 'semana' : 'semanas'}`
  if (months < 12) return `hace ${months} ${months === 1 ? 'mes' : 'meses'}`
  return `hace ${years} ${years === 1 ? 'año' : 'años'}`
}

/**
 * Format date to string
 */
export function formatDate(
  date: Date | number,
  format: 'short' | 'long' | 'time' | 'datetime' | 'iso' = 'short'
): string {
  const d = date instanceof Date ? date : new Date(date)

  switch (format) {
    case 'short':
      return d.toLocaleDateString('es-ES', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
    case 'long':
      return d.toLocaleDateString('es-ES', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    case 'time':
      return d.toLocaleTimeString('es-ES', {
        hour: '2-digit',
        minute: '2-digit',
      })
    case 'datetime':
      return d.toLocaleString('es-ES', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    case 'iso':
      return d.toISOString()
    default:
      return d.toString()
  }
}

/**
 * Check if date is today
 */
export function isToday(date: Date | number): boolean {
  const d = date instanceof Date ? date : new Date(date)
  const today = new Date()
  return (
    d.getDate() === today.getDate() &&
    d.getMonth() === today.getMonth() &&
    d.getFullYear() === today.getFullYear()
  )
}

/**
 * Check if date is yesterday
 */
export function isYesterday(date: Date | number): boolean {
  const d = date instanceof Date ? date : new Date(date)
  const yesterday = new Date()
  yesterday.setDate(yesterday.getDate() - 1)
  return (
    d.getDate() === yesterday.getDate() &&
    d.getMonth() === yesterday.getMonth() &&
    d.getFullYear() === yesterday.getFullYear()
  )
}

/**
 * Check if date is this week
 */
export function isThisWeek(date: Date | number): boolean {
  const d = date instanceof Date ? date : new Date(date)
  const now = new Date()
  const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
  return d >= weekAgo && d <= now
}

/**
 * Get start of day
 */
export function startOfDay(date: Date | number): Date {
  const d = date instanceof Date ? new Date(date) : new Date(date)
  d.setHours(0, 0, 0, 0)
  return d
}

/**
 * Get end of day
 */
export function endOfDay(date: Date | number): Date {
  const d = date instanceof Date ? new Date(date) : new Date(date)
  d.setHours(23, 59, 59, 999)
  return d
}

/**
 * Get start of week
 */
export function startOfWeek(date: Date | number, weekStartsOn: number = 1): Date {
  const d = date instanceof Date ? new Date(date) : new Date(date)
  const day = d.getDay()
  const diff = (day < weekStartsOn ? 7 : 0) + day - weekStartsOn
  d.setDate(d.getDate() - diff)
  return startOfDay(d)
}

/**
 * Get end of week
 */
export function endOfWeek(date: Date | number, weekStartsOn: number = 1): Date {
  const start = startOfWeek(date, weekStartsOn)
  start.setDate(start.getDate() + 6)
  return endOfDay(start)
}

/**
 * Get date range for period
 */
export function getDateRange(period: 'today' | 'yesterday' | 'thisWeek' | 'thisMonth' | 'thisYear'): {
  start: Date
  end: Date
} {
  const now = new Date()

  switch (period) {
    case 'today':
      return { start: startOfDay(now), end: endOfDay(now) }
    case 'yesterday':
      const yesterday = new Date(now)
      yesterday.setDate(yesterday.getDate() - 1)
      return { start: startOfDay(yesterday), end: endOfDay(yesterday) }
    case 'thisWeek':
      return { start: startOfWeek(now), end: endOfWeek(now) }
    case 'thisMonth':
      return {
        start: new Date(now.getFullYear(), now.getMonth(), 1),
        end: new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59, 999),
      }
    case 'thisYear':
      return {
        start: new Date(now.getFullYear(), 0, 1),
        end: new Date(now.getFullYear(), 11, 31, 23, 59, 59, 999),
      }
  }
}

/**
 * Format duration
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`
  }

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)

  if (minutes < 60) {
    return `${minutes}m ${remainingSeconds}s`
  }

  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60

  if (hours < 24) {
    return `${hours}h ${remainingMinutes}m ${remainingSeconds}s`
  }

  const days = Math.floor(hours / 24)
  const remainingHours = hours % 24

  return `${days}d ${remainingHours}h ${remainingMinutes}m`
}

/**
 * Parse date string
 */
export function parseDate(dateString: string): Date | null {
  const date = new Date(dateString)
  return isNaN(date.getTime()) ? null : date
}

/**
 * Get timezone offset in minutes
 */
export function getTimezoneOffset(): number {
  return new Date().getTimezoneOffset()
}

/**
 * Convert to UTC
 */
export function toUTC(date: Date | number): Date {
  const d = date instanceof Date ? new Date(date) : new Date(date)
  return new Date(d.getTime() + d.getTimezoneOffset() * 60000)
}




