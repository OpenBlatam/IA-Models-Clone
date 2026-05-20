/**
 * Utilidades de Tiempo
 * ===================
 * 
 * Funciones para manejo de tiempo, delays, timers y scheduling
 */

/**
 * Delay asíncrono
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Timeout con Promise
 */
export function timeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
    )
  ])
}

/**
 * Retry con backoff exponencial
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number
    initialDelay?: number
    maxDelay?: number
    backoffFactor?: number
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    backoffFactor = 2
  } = options

  let lastError: Error | null = null
  let delay = initialDelay

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error as Error
      
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, delay))
        delay = Math.min(delay * backoffFactor, maxDelay)
      }
    }
  }

  throw lastError || new Error('Retry failed')
}

/**
 * Debounce mejorado
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  immediate: boolean = false
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null
      if (!immediate) func(...args)
    }

    const callNow = immediate && !timeout

    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(later, wait)

    if (callNow) func(...args)
  }
}

/**
 * Throttle mejorado
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean

  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

/**
 * Scheduler de tareas
 */
export class TaskScheduler {
  private tasks: Map<string, NodeJS.Timeout> = new Map()

  /**
   * Programa una tarea
   */
  schedule(id: string, task: () => void, delayMs: number): void {
    this.cancel(id)
    const timeout = setTimeout(() => {
      task()
      this.tasks.delete(id)
    }, delayMs)
    this.tasks.set(id, timeout)
  }

  /**
   * Cancela una tarea programada
   */
  cancel(id: string): void {
    const timeout = this.tasks.get(id)
    if (timeout) {
      clearTimeout(timeout)
      this.tasks.delete(id)
    }
  }

  /**
   * Cancela todas las tareas
   */
  cancelAll(): void {
    this.tasks.forEach(timeout => clearTimeout(timeout))
    this.tasks.clear()
  }

  /**
   * Verifica si una tarea está programada
   */
  hasTask(id: string): boolean {
    return this.tasks.has(id)
  }
}

/**
 * Formatea duración en milisegundos a string legible
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
  
  const hours = Math.floor(ms / 3600000)
  const minutes = Math.floor((ms % 3600000) / 60000)
  const seconds = Math.floor((ms % 60000) / 1000)
  
  return `${hours}h ${minutes}m ${seconds}s`
}

/**
 * Formatea tiempo relativo (hace X tiempo)
 */
export function formatRelativeTime(date: Date | number): string {
  const now = Date.now()
  const then = typeof date === 'number' ? date : date.getTime()
  const diff = now - then

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
 * Verifica si una fecha está en el pasado
 */
export function isPast(date: Date | number): boolean {
  const then = typeof date === 'number' ? date : date.getTime()
  return then < Date.now()
}

/**
 * Verifica si una fecha está en el futuro
 */
export function isFuture(date: Date | number): boolean {
  const then = typeof date === 'number' ? date : date.getTime()
  return then > Date.now()
}

/**
 * Obtiene el tiempo hasta una fecha
 */
export function getTimeUntil(date: Date | number): number {
  const then = typeof date === 'number' ? date : date.getTime()
  return Math.max(0, then - Date.now())
}

/**
 * Obtiene el tiempo desde una fecha
 */
export function getTimeSince(date: Date | number): number {
  const then = typeof date === 'number' ? date : date.getTime()
  return Math.max(0, Date.now() - then)
}

/**
 * Crea un intervalo que se limpia automáticamente
 */
export function createAutoClearingInterval(
  callback: () => void,
  delay: number
): () => void {
  const interval = setInterval(callback, delay)
  return () => clearInterval(interval)
}

/**
 * Crea un timeout que se limpia automáticamente
 */
export function createAutoClearingTimeout(
  callback: () => void,
  delay: number
): () => void {
  const timeout = setTimeout(callback, delay)
  return () => clearTimeout(timeout)
}






