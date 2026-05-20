/**
 * Performance Metrics System for TruthGPT
 * Tracks and analyzes performance metrics in real-time
 */

interface MetricEntry {
  timestamp: number
  duration: number
  success: boolean
  operation: string
  metadata?: Record<string, any>
}

interface AggregatedMetrics {
  totalOperations: number
  successfulOperations: number
  failedOperations: number
  successRate: number
  avgDuration: number
  minDuration: number
  maxDuration: number
  p95Duration: number
  p99Duration: number
  operationsPerSecond: number
  lastUpdated: number
}

export class PerformanceMetrics {
  private metrics: Map<string, MetricEntry[]> = new Map()
  private windowSize: number // milliseconds
  private startTime: number

  constructor(windowSize: number = 60000) { // 1 minute default
    this.windowSize = windowSize
    this.startTime = Date.now()
  }

  /**
   * Record a metric
   */
  record(
    operation: string,
    duration: number,
    success: boolean,
    metadata?: Record<string, any>
  ): void {
    const entry: MetricEntry = {
      timestamp: Date.now(),
      duration,
      success,
      operation,
      metadata
    }

    if (!this.metrics.has(operation)) {
      this.metrics.set(operation, [])
    }

    this.metrics.get(operation)!.push(entry)

    // Cleanup old entries
    this.cleanup(operation)
  }

  /**
   * Get aggregated metrics for an operation
   */
  getMetrics(operation: string): AggregatedMetrics | null {
    const entries = this.metrics.get(operation)
    if (!entries || entries.length === 0) {
      return null
    }

    const now = Date.now()
    const cutoff = now - this.windowSize
    const recentEntries = entries.filter(e => e.timestamp > cutoff)

    if (recentEntries.length === 0) {
      return null
    }

    const durations = recentEntries.map(e => e.duration).sort((a, b) => a - b)
    const successful = recentEntries.filter(e => e.success).length
    const total = recentEntries.length

    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length
    const windowSeconds = this.windowSize / 1000
    const opsPerSecond = total / windowSeconds

    return {
      totalOperations: total,
      successfulOperations: successful,
      failedOperations: total - successful,
      successRate: (successful / total) * 100,
      avgDuration: Math.round(avgDuration * 100) / 100,
      minDuration: durations[0],
      maxDuration: durations[durations.length - 1],
      p95Duration: this.percentile(durations, 0.95),
      p99Duration: this.percentile(durations, 0.99),
      operationsPerSecond: Math.round(opsPerSecond * 100) / 100,
      lastUpdated: now
    }
  }

  /**
   * Get all metrics
   */
  getAllMetrics(): Record<string, AggregatedMetrics> {
    const result: Record<string, AggregatedMetrics> = {}
    
    for (const operation of this.metrics.keys()) {
      const metrics = this.getMetrics(operation)
      if (metrics) {
        result[operation] = metrics
      }
    }

    return result
  }

  /**
   * Get performance summary
   */
  getSummary(): {
    uptimeSeconds: number
    totalOperations: number
    overallSuccessRate: number
    avgDuration: number
    operations: Record<string, AggregatedMetrics>
  } {
    const allMetrics = this.getAllMetrics()
    let totalOps = 0
    let totalSuccess = 0
    let totalDuration = 0
    let count = 0

    for (const metrics of Object.values(allMetrics)) {
      totalOps += metrics.totalOperations
      totalSuccess += metrics.successfulOperations
      totalDuration += metrics.avgDuration * metrics.totalOperations
      count += metrics.totalOperations
    }

    const uptime = (Date.now() - this.startTime) / 1000
    const overallSuccessRate = totalOps > 0 ? (totalSuccess / totalOps) * 100 : 100
    const avgDuration = count > 0 ? totalDuration / count : 0

    return {
      uptimeSeconds: Math.round(uptime),
      totalOperations: totalOps,
      overallSuccessRate: Math.round(overallSuccessRate * 100) / 100,
      avgDuration: Math.round(avgDuration * 100) / 100,
      operations: allMetrics
    }
  }

  /**
   * Cleanup old entries for an operation
   */
  private cleanup(operation: string): void {
    const entries = this.metrics.get(operation)
    if (!entries) return

    const cutoff = Date.now() - this.windowSize
    const filtered = entries.filter(e => e.timestamp > cutoff)
    
    if (filtered.length !== entries.length) {
      this.metrics.set(operation, filtered)
    }
  }

  /**
   * Calculate percentile
   */
  private percentile(sortedArray: number[], percentile: number): number {
    const index = Math.ceil(sortedArray.length * percentile) - 1
    return sortedArray[Math.max(0, Math.min(index, sortedArray.length - 1))]
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.metrics.clear()
    this.startTime = Date.now()
  }
}

// Global metrics instance
export const performanceMetrics = new PerformanceMetrics()


