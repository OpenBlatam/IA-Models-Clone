/**
 * Advanced Statistics
 * Sistema de estadísticas avanzadas con análisis profundo
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export interface AdvancedStats {
  overview: {
    totalModels: number
    successful: number
    failed: number
    successRate: number
    totalDuration: number
    avgDuration: number
    medianDuration: number
  }
  trends: {
    daily: Array<{ date: string; count: number; successRate: number }>
    weekly: Array<{ week: string; count: number; successRate: number }>
    monthly: Array<{ month: string; count: number; successRate: number }>
  }
  performance: {
    fastest: ProactiveBuildResult | null
    slowest: ProactiveBuildResult | null
    average: number
    percentile25: number
    percentile50: number
    percentile75: number
    percentile95: number
  }
  patterns: {
    bestTimeOfDay: string
    bestDayOfWeek: string
    commonErrors: Array<{ error: string; count: number }>
    popularCategories: Array<{ category: string; count: number }>
  }
  predictions: {
    nextBuildEstimate: number
    successProbability: number
    recommendedBatchSize: number
  }
}

export class AdvancedStatistics {
  /**
   * Calcular estadísticas avanzadas
   */
  calculateAdvancedStats(models: ProactiveBuildResult[]): AdvancedStats {
    const overview = this.calculateOverview(models)
    const trends = this.calculateTrends(models)
    const performance = this.calculatePerformance(models)
    const patterns = this.calculatePatterns(models)
    const predictions = this.calculatePredictions(models, overview, performance)

    return {
      overview,
      trends,
      performance,
      patterns,
      predictions,
    }
  }

  /**
   * Calcular resumen general
   */
  private calculateOverview(models: ProactiveBuildResult[]): AdvancedStats['overview'] {
    const totalModels = models.length
    const successful = models.filter(m => m.status === 'completed').length
    const failed = totalModels - successful
    const successRate = totalModels > 0 ? successful / totalModels : 0

    const durations = models
      .filter(m => m.duration)
      .map(m => m.duration!)
      .sort((a, b) => a - b)

    const totalDuration = durations.reduce((sum, d) => sum + d, 0)
    const avgDuration = durations.length > 0 ? totalDuration / durations.length : 0
    const medianDuration = this.calculateMedian(durations)

    return {
      totalModels,
      successful,
      failed,
      successRate,
      totalDuration,
      avgDuration,
      medianDuration,
    }
  }

  /**
   * Calcular tendencias
   */
  private calculateTrends(models: ProactiveBuildResult[]): AdvancedStats['trends'] {
    const daily = this.calculateDailyTrends(models)
    const weekly = this.calculateWeeklyTrends(models)
    const monthly = this.calculateMonthlyTrends(models)

    return { daily, weekly, monthly }
  }

  /**
   * Tendencias diarias
   */
  private calculateDailyTrends(models: ProactiveBuildResult[]): Array<{ date: string; count: number; successRate: number }> {
    const dailyMap = new Map<string, { total: number; successful: number }>()

    models.forEach(model => {
      const date = new Date(model.endTime || model.startTime || Date.now())
        .toISOString()
        .split('T')[0]

      const existing = dailyMap.get(date) || { total: 0, successful: 0 }
      existing.total++
      if (model.status === 'completed') existing.successful++
      dailyMap.set(date, existing)
    })

    return Array.from(dailyMap.entries())
      .map(([date, data]) => ({
        date,
        count: data.total,
        successRate: data.total > 0 ? data.successful / data.total : 0,
      }))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-30) // Últimos 30 días
  }

  /**
   * Tendencias semanales
   */
  private calculateWeeklyTrends(models: ProactiveBuildResult[]): Array<{ week: string; count: number; successRate: number }> {
    const weeklyMap = new Map<string, { total: number; successful: number }>()

    models.forEach(model => {
      const date = new Date(model.endTime || model.startTime || Date.now())
      const weekStart = new Date(date)
      weekStart.setDate(date.getDate() - date.getDay())
      const week = weekStart.toISOString().split('T')[0]

      const existing = weeklyMap.get(week) || { total: 0, successful: 0 }
      existing.total++
      if (model.status === 'completed') existing.successful++
      weeklyMap.set(week, existing)
    })

    return Array.from(weeklyMap.entries())
      .map(([week, data]) => ({
        week,
        count: data.total,
        successRate: data.total > 0 ? data.successful / data.total : 0,
      }))
      .sort((a, b) => a.week.localeCompare(b.week))
      .slice(-12) // Últimas 12 semanas
  }

  /**
   * Tendencias mensuales
   */
  private calculateMonthlyTrends(models: ProactiveBuildResult[]): Array<{ month: string; count: number; successRate: number }> {
    const monthlyMap = new Map<string, { total: number; successful: number }>()

    models.forEach(model => {
      const date = new Date(model.endTime || model.startTime || Date.now())
      const month = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`

      const existing = monthlyMap.get(month) || { total: 0, successful: 0 }
      existing.total++
      if (model.status === 'completed') existing.successful++
      monthlyMap.set(month, existing)
    })

    return Array.from(monthlyMap.entries())
      .map(([month, data]) => ({
        month,
        count: data.total,
        successRate: data.total > 0 ? data.successful / data.total : 0,
      }))
      .sort((a, b) => a.month.localeCompare(b.month))
      .slice(-12) // Últimos 12 meses
  }

  /**
   * Calcular performance
   */
  private calculatePerformance(models: ProactiveBuildResult[]): AdvancedStats['performance'] {
    const durations = models
      .filter(m => m.duration)
      .map(m => m.duration!)
      .sort((a, b) => a - b)

    if (durations.length === 0) {
      return {
        fastest: null,
        slowest: null,
        average: 0,
        percentile25: 0,
        percentile50: 0,
        percentile75: 0,
        percentile95: 0,
      }
    }

    const fastestModel = models.find(m => m.duration === durations[0])
    const slowestModel = models.find(m => m.duration === durations[durations.length - 1])

    return {
      fastest: fastestModel || null,
      slowest: slowestModel || null,
      average: durations.reduce((sum, d) => sum + d, 0) / durations.length,
      percentile25: this.calculatePercentile(durations, 25),
      percentile50: this.calculatePercentile(durations, 50),
      percentile75: this.calculatePercentile(durations, 75),
      percentile95: this.calculatePercentile(durations, 95),
    }
  }

  /**
   * Calcular patrones
   */
  private calculatePatterns(models: ProactiveBuildResult[]): AdvancedStats['patterns'] {
    const timeOfDayMap = new Map<number, { total: number; successful: number }>()
    const dayOfWeekMap = new Map<number, { total: number; successful: number }>()
    const errorMap = new Map<string, number>()
    const categoryMap = new Map<string, number>()

    models.forEach(model => {
      const date = new Date(model.endTime || model.startTime || Date.now())
      const hour = date.getHours()
      const day = date.getDay()

      // Tiempo del día
      const timeData = timeOfDayMap.get(hour) || { total: 0, successful: 0 }
      timeData.total++
      if (model.status === 'completed') timeData.successful++
      timeOfDayMap.set(hour, timeData)

      // Día de la semana
      const dayData = dayOfWeekMap.get(day) || { total: 0, successful: 0 }
      dayData.total++
      if (model.status === 'completed') dayData.successful++
      dayOfWeekMap.set(day, dayData)

      // Errores
      if (model.error) {
        errorMap.set(model.error, (errorMap.get(model.error) || 0) + 1)
      }

      // Categorías (si están disponibles)
      // categoryMap logic would go here
    })

    // Mejor tiempo del día
    let bestTime = 12
    let bestTimeRate = 0
    timeOfDayMap.forEach((data, hour) => {
      const rate = data.total > 0 ? data.successful / data.total : 0
      if (rate > bestTimeRate) {
        bestTimeRate = rate
        bestTime = hour
      }
    })

    // Mejor día de la semana
    let bestDay = 1
    let bestDayRate = 0
    dayOfWeekMap.forEach((data, day) => {
      const rate = data.total > 0 ? data.successful / data.total : 0
      if (rate > bestDayRate) {
        bestDayRate = rate
        bestDay = day
      }
    })

    const dayNames = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']

    return {
      bestTimeOfDay: `${bestTime}:00`,
      bestDayOfWeek: dayNames[bestDay],
      commonErrors: Array.from(errorMap.entries())
        .map(([error, count]) => ({ error, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
      popularCategories: Array.from(categoryMap.entries())
        .map(([category, count]) => ({ category, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
    }
  }

  /**
   * Calcular predicciones
   */
  private calculatePredictions(
    models: ProactiveBuildResult[],
    overview: AdvancedStats['overview'],
    performance: AdvancedStats['performance']
  ): AdvancedStats['predictions'] {
    const recentModels = models.slice(-10)
    const recentAvg = recentModels
      .filter(m => m.duration)
      .reduce((sum, m) => sum + (m.duration || 0), 0) / (recentModels.length || 1)

    return {
      nextBuildEstimate: recentAvg || performance.average,
      successProbability: overview.successRate,
      recommendedBatchSize: Math.min(Math.max(Math.ceil(overview.totalModels / 10), 1), 10),
    }
  }

  /**
   * Calcular mediana
   */
  private calculateMedian(values: number[]): number {
    if (values.length === 0) return 0
    const sorted = [...values].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid]
  }

  /**
   * Calcular percentil
   */
  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0
    const sorted = [...values].sort((a, b) => a - b)
    const index = Math.ceil((percentile / 100) * sorted.length) - 1
    return sorted[Math.max(0, Math.min(index, sorted.length - 1))]
  }

  /**
   * Exportar estadísticas
   */
  exportStats(stats: AdvancedStats): string {
    return JSON.stringify(stats, null, 2)
  }
}

// Singleton instance
let advancedStatisticsInstance: AdvancedStatistics | null = null

export function getAdvancedStatistics(): AdvancedStatistics {
  if (!advancedStatisticsInstance) {
    advancedStatisticsInstance = new AdvancedStatistics()
  }
  return advancedStatisticsInstance
}

export default AdvancedStatistics










