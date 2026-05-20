/**
 * Intelligent Alerts
 * Sistema de alertas inteligentes
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export type AlertSeverity = 'info' | 'warning' | 'error' | 'critical'

export interface AlertRule {
  id: string
  name: string
  condition: (data: any) => boolean
  severity: AlertSeverity
  message: string | ((data: any) => string)
  enabled: boolean
  cooldown?: number // Tiempo mínimo entre alertas (ms)
  lastTriggered?: number
}

export interface Alert {
  id: string
  ruleId: string
  severity: AlertSeverity
  message: string
  timestamp: number
  data?: any
  acknowledged: boolean
}

export class IntelligentAlerts {
  private rules: Map<string, AlertRule> = new Map()
  private alerts: Alert[] = []
  private callbacks: Set<(alert: Alert) => void> = new Set()
  private maxAlerts: number = 1000

  /**
   * Registrar regla de alerta
   */
  registerRule(rule: AlertRule): void {
    this.rules.set(rule.id, {
      ...rule,
      enabled: rule.enabled !== false,
    })
  }

  /**
   * Eliminar regla
   */
  unregisterRule(id: string): void {
    this.rules.delete(id)
  }

  /**
   * Evaluar datos y generar alertas
   */
  evaluate(data: any): Alert[] {
    const newAlerts: Alert[] = []

    this.rules.forEach(rule => {
      if (!rule.enabled) return

      // Verificar cooldown
      if (rule.cooldown && rule.lastTriggered) {
        const timeSinceLastTrigger = Date.now() - rule.lastTriggered
        if (timeSinceLastTrigger < rule.cooldown) {
          return
        }
      }

      // Evaluar condición
      try {
        if (rule.condition(data)) {
          const message =
            typeof rule.message === 'function'
              ? rule.message(data)
              : rule.message

          const alert: Alert = {
            id: `alert-${Date.now()}-${Math.random()}`,
            ruleId: rule.id,
            severity: rule.severity,
            message,
            timestamp: Date.now(),
            data,
            acknowledged: false,
          }

          newAlerts.push(alert)
          this.alerts.unshift(alert)
          this.alerts = this.alerts.slice(0, this.maxAlerts)

          rule.lastTriggered = Date.now()

          // Notificar callbacks
          this.callbacks.forEach(callback => callback(alert))
        }
      } catch (error) {
        console.error(`Error evaluating rule ${rule.id}:`, error)
      }
    })

    return newAlerts
  }

  /**
   * Suscribirse a alertas
   */
  subscribe(callback: (alert: Alert) => void): () => void {
    this.callbacks.add(callback)
    return () => {
      this.callbacks.delete(callback)
    }
  }

  /**
   * Obtener alertas
   */
  getAlerts(limit?: number): Alert[] {
    return limit ? this.alerts.slice(0, limit) : [...this.alerts]
  }

  /**
   * Obtener alertas no reconocidas
   */
  getUnacknowledgedAlerts(): Alert[] {
    return this.alerts.filter(a => !a.acknowledged)
  }

  /**
   * Reconocer alerta
   */
  acknowledge(id: string): void {
    const alert = this.alerts.find(a => a.id === id)
    if (alert) {
      alert.acknowledged = true
    }
  }

  /**
   * Reconocer todas las alertas
   */
  acknowledgeAll(): void {
    this.alerts.forEach(alert => {
      alert.acknowledged = true
    })
  }

  /**
   * Eliminar alerta
   */
  removeAlert(id: string): void {
    this.alerts = this.alerts.filter(a => a.id !== id)
  }

  /**
   * Limpiar alertas
   */
  clearAlerts(): void {
    this.alerts = []
  }

  /**
   * Obtener estadísticas
   */
  getStats(): {
    total: number
    unacknowledged: number
    bySeverity: Record<AlertSeverity, number>
    recentCount: number
  } {
    const bySeverity: Record<AlertSeverity, number> = {
      info: 0,
      warning: 0,
      error: 0,
      critical: 0,
    }

    this.alerts.forEach(alert => {
      bySeverity[alert.severity]++
    })

    const oneHourAgo = Date.now() - 3600000
    const recentCount = this.alerts.filter(
      a => a.timestamp > oneHourAgo
    ).length

    return {
      total: this.alerts.length,
      unacknowledged: this.alerts.filter(a => !a.acknowledged).length,
      bySeverity,
      recentCount,
    }
  }

  /**
   * Inicializar reglas predefinidas
   */
  initializeDefaultRules(): void {
    // Alta tasa de fallos
    this.registerRule({
      id: 'high-failure-rate',
      name: 'Alta Tasa de Fallos',
      condition: (data: { models: ProactiveBuildResult[] }) => {
        if (!data.models || data.models.length < 5) return false
        const recent = data.models.slice(-10)
        const failureRate =
          recent.filter(m => m.status === 'failed').length / recent.length
        return failureRate > 0.5
      },
      severity: 'error',
      message: (data) => {
        const recent = data.models.slice(-10)
        const failures = recent.filter(m => m.status === 'failed').length
        return `Alta tasa de fallos: ${failures}/10 modelos fallaron`
      },
      enabled: true,
      cooldown: 300000, // 5 minutos
    })

    // Cola muy larga
    this.registerRule({
      id: 'long-queue',
      name: 'Cola Muy Larga',
      condition: (data: { queueLength: number }) => {
        return data.queueLength > 20
      },
      severity: 'warning',
      message: (data) => `Cola muy larga: ${data.queueLength} modelos pendientes`,
      enabled: true,
      cooldown: 600000, // 10 minutos
    })

    // Duración muy larga
    this.registerRule({
      id: 'long-duration',
      name: 'Duración Muy Larga',
      condition: (data: { models: ProactiveBuildResult[] }) => {
        if (!data.models || data.models.length === 0) return false
        const recent = data.models.slice(-5)
        const avgDuration =
          recent
            .filter(m => m.duration)
            .reduce((sum, m) => sum + (m.duration || 0), 0) /
          recent.length
        return avgDuration > 120000 // 2 minutos
      },
      severity: 'warning',
      message: () => 'Los modelos están tomando más tiempo del esperado',
      enabled: true,
      cooldown: 600000,
    })

    // Sin actividad reciente
    this.registerRule({
      id: 'no-recent-activity',
      name: 'Sin Actividad Reciente',
      condition: (data: { models: ProactiveBuildResult[]; queueLength: number }) => {
        if (!data.models || data.models.length === 0) return false
        if (data.queueLength > 0) return false

        const lastModel = data.models[0]
        const timeSinceLastModel =
          Date.now() - (lastModel.endTime || lastModel.startTime || Date.now())
        return timeSinceLastModel > 3600000 // 1 hora
      },
      severity: 'info',
      message: () => 'No hay actividad reciente en la construcción de modelos',
      enabled: true,
      cooldown: 1800000, // 30 minutos
    })
  }

  /**
   * Limpiar reglas
   */
  clear(): void {
    this.rules.clear()
    this.alerts = []
    this.callbacks.clear()
  }
}

// Singleton instance
let intelligentAlertsInstance: IntelligentAlerts | null = null

export function getIntelligentAlerts(): IntelligentAlerts {
  if (!intelligentAlertsInstance) {
    intelligentAlertsInstance = new IntelligentAlerts()
    intelligentAlertsInstance.initializeDefaultRules()
  }
  return intelligentAlertsInstance
}

export default IntelligentAlerts










