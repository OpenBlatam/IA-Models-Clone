/**
 * Performance monitoring utilities
 */

export interface PerformanceMetric {
  name: string
  startTime: number
  endTime?: number
  duration?: number
  metadata?: Record<string, any>
}

class PerformanceMonitor {
  private metrics: Map<string, PerformanceMetric> = new Map()
  private history: PerformanceMetric[] = []
  private maxHistory = 1000

  start(name: string, metadata?: Record<string, any>): string {
    const id = `${name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    this.metrics.set(id, {
      name,
      startTime: performance.now(),
      metadata,
    })
    return id
  }

  end(id: string, metadata?: Record<string, any>): PerformanceMetric | null {
    const metric = this.metrics.get(id)
    if (!metric) return null

    metric.endTime = performance.now()
    metric.duration = metric.endTime - metric.startTime
    if (metadata) {
      metric.metadata = { ...metric.metadata, ...metadata }
    }

    this.metrics.delete(id)
    this.history.push(metric)

    // Keep only last maxHistory
    if (this.history.length > this.maxHistory) {
      this.history.shift()
    }

    return metric
  }

  getMetric(id: string): PerformanceMetric | undefined {
    return this.metrics.get(id)
  }

  getHistory(name?: string, limit?: number): PerformanceMetric[] {
    let filtered = this.history
    if (name) {
      filtered = filtered.filter(m => m.name === name)
    }
    if (limit) {
      filtered = filtered.slice(-limit)
    }
    return filtered
  }

  getAverageDuration(name: string): number {
    const metrics = this.getHistory(name)
    if (metrics.length === 0) return 0

    const total = metrics.reduce((sum, m) => sum + (m.duration || 0), 0)
    return total / metrics.length
  }

  clear(): void {
    this.metrics.clear()
    this.history = []
  }

  export(): any {
    return {
      active: Array.from(this.metrics.values()),
      history: this.history,
      averages: this.getAverageDurations(),
    }
  }

  private getAverageDurations(): Record<string, number> {
    const names = new Set(this.history.map(m => m.name))
    const averages: Record<string, number> = {}
    for (const name of names) {
      averages[name] = this.getAverageDuration(name)
    }
    return averages
  }
}

export const performanceMonitor = new PerformanceMonitor()


