/**
 * Utilidades de Monitoreo de Performance
 * =======================================
 * 
 * Funciones para monitorear y medir performance
 */

export interface PerformanceMetric {
  name: string
  duration: number
  timestamp: number
  metadata?: Record<string, any>
}

export interface PerformanceReport {
  metrics: PerformanceMetric[]
  totalDuration: number
  averageDuration: number
  minDuration: number
  maxDuration: number
  count: number
}

/**
 * Clase para monitorear performance
 */
export class PerformanceMonitor {
  private metrics: PerformanceMetric[] = []
  private maxMetrics: number = 1000

  /**
   * Mide el tiempo de ejecución de una función
   */
  async measure<T>(
    name: string,
    fn: () => T | Promise<T>,
    metadata?: Record<string, any>
  ): Promise<T> {
    const start = performance.now()
    
    try {
      const result = await fn()
      const duration = performance.now() - start
      
      this.recordMetric({
        name,
        duration,
        timestamp: Date.now(),
        metadata
      })
      
      return result
    } catch (error) {
      const duration = performance.now() - start
      
      this.recordMetric({
        name,
        duration,
        timestamp: Date.now(),
        metadata: {
          ...metadata,
          error: error instanceof Error ? error.message : String(error)
        }
      })
      
      throw error
    }
  }

  /**
   * Registra una métrica
   */
  recordMetric(metric: PerformanceMetric): void {
    this.metrics.push(metric)
    
    if (this.metrics.length > this.maxMetrics) {
      this.metrics.shift()
    }
  }

  /**
   * Obtiene métricas por nombre
   */
  getMetrics(name?: string): PerformanceMetric[] {
    if (name) {
      return this.metrics.filter(m => m.name === name)
    }
    return [...this.metrics]
  }

  /**
   * Genera un reporte de performance
   */
  getReport(name?: string): PerformanceReport {
    const metrics = name ? this.getMetrics(name) : this.metrics
    
    if (metrics.length === 0) {
      return {
        metrics: [],
        totalDuration: 0,
        averageDuration: 0,
        minDuration: 0,
        maxDuration: 0,
        count: 0
      }
    }

    const durations = metrics.map(m => m.duration)
    const totalDuration = durations.reduce((sum, d) => sum + d, 0)
    const averageDuration = totalDuration / durations.length
    const minDuration = Math.min(...durations)
    const maxDuration = Math.max(...durations)

    return {
      metrics: [...metrics],
      totalDuration,
      averageDuration,
      minDuration,
      maxDuration,
      count: metrics.length
    }
  }

  /**
   * Limpia las métricas
   */
  clear(): void {
    this.metrics = []
  }

  /**
   * Obtiene estadísticas por nombre de métrica
   */
  getStatsByName(): Record<string, PerformanceReport> {
    const names = new Set(this.metrics.map(m => m.name))
    const stats: Record<string, PerformanceReport> = {}

    for (const name of names) {
      stats[name] = this.getReport(name)
    }

    return stats
  }
}

/**
 * Monitor global
 */
let globalMonitor: PerformanceMonitor | null = null

/**
 * Obtiene el monitor global
 */
export function getPerformanceMonitor(): PerformanceMonitor {
  if (!globalMonitor) {
    globalMonitor = new PerformanceMonitor()
  }
  return globalMonitor
}

/**
 * Mide el tiempo de ejecución (función helper)
 */
export async function measurePerformance<T>(
  name: string,
  fn: () => T | Promise<T>,
  metadata?: Record<string, any>
): Promise<T> {
  return getPerformanceMonitor().measure(name, fn, metadata)
}

/**
 * Mide el tiempo de ejecución síncrono
 */
export function measurePerformanceSync<T>(
  name: string,
  fn: () => T,
  metadata?: Record<string, any>
): T {
  const start = performance.now()
  
  try {
    const result = fn()
    const duration = performance.now() - start
    
    getPerformanceMonitor().recordMetric({
      name,
      duration,
      timestamp: Date.now(),
      metadata
    })
    
    return result
  } catch (error) {
    const duration = performance.now() - start
    
    getPerformanceMonitor().recordMetric({
      name,
      duration,
      timestamp: Date.now(),
      metadata: {
        ...metadata,
        error: error instanceof Error ? error.message : String(error)
      }
    })
    
    throw error
  }
}

/**
 * Crea un decorador para medir performance
 */
export function measure(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value

  descriptor.value = async function (...args: any[]) {
    return measurePerformance(
      `${target.constructor.name}.${propertyKey}`,
      () => originalMethod.apply(this, args)
    )
  }

  return descriptor
}






