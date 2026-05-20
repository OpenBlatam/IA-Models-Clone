/**
 * Utilidades de Analytics
 * =======================
 * 
 * Funciones para tracking y analytics
 */

export interface AnalyticsEvent {
  name: string
  properties?: Record<string, any>
  timestamp: number
  userId?: string
  sessionId?: string
}

export interface AnalyticsConfig {
  enabled?: boolean
  debug?: boolean
  userId?: string
  sessionId?: string
  onEvent?: (event: AnalyticsEvent) => void
}

/**
 * Clase para manejar analytics
 */
export class Analytics {
  private config: AnalyticsConfig
  private events: AnalyticsEvent[] = []
  private maxEvents: number = 1000

  constructor(config: AnalyticsConfig = {}) {
    this.config = {
      enabled: true,
      debug: false,
      ...config
    }
  }

  /**
   * Registra un evento
   */
  track(eventName: string, properties?: Record<string, any>): void {
    if (!this.config.enabled) return

    const event: AnalyticsEvent = {
      name: eventName,
      properties,
      timestamp: Date.now(),
      userId: this.config.userId,
      sessionId: this.config.sessionId
    }

    this.events.push(event)

    if (this.events.length > this.maxEvents) {
      this.events.shift()
    }

    if (this.config.debug) {
      console.log('[Analytics]', event)
    }

    if (this.config.onEvent) {
      this.config.onEvent(event)
    }
  }

  /**
   * Registra un evento de página vista
   */
  page(pageName: string, properties?: Record<string, any>): void {
    this.track('page_view', {
      page: pageName,
      ...properties
    })
  }

  /**
   * Registra un evento de click
   */
  click(elementName: string, properties?: Record<string, any>): void {
    this.track('click', {
      element: elementName,
      ...properties
    })
  }

  /**
   * Registra un evento de conversión
   */
  conversion(conversionName: string, value?: number, properties?: Record<string, any>): void {
    this.track('conversion', {
      conversion: conversionName,
      value,
      ...properties
    })
  }

  /**
   * Registra un error
   */
  error(error: Error | string, properties?: Record<string, any>): void {
    this.track('error', {
      error: error instanceof Error ? error.message : error,
      errorStack: error instanceof Error ? error.stack : undefined,
      ...properties
    })
  }

  /**
   * Establece el usuario
   */
  setUser(userId: string): void {
    this.config.userId = userId
  }

  /**
   * Establece la sesión
   */
  setSession(sessionId: string): void {
    this.config.sessionId = sessionId
  }

  /**
   * Obtiene los eventos
   */
  getEvents(eventName?: string): AnalyticsEvent[] {
    if (eventName) {
      return this.events.filter(e => e.name === eventName)
    }
    return [...this.events]
  }

  /**
   * Limpia los eventos
   */
  clearEvents(): void {
    this.events = []
  }

  /**
   * Exporta los eventos
   */
  exportEvents(): string {
    return JSON.stringify(this.events, null, 2)
  }

  /**
   * Obtiene estadísticas de eventos
   */
  getStats(): Record<string, number> {
    const stats: Record<string, number> = {}

    for (const event of this.events) {
      stats[event.name] = (stats[event.name] || 0) + 1
    }

    return stats
  }
}

/**
 * Analytics global
 */
let globalAnalytics: Analytics | null = null

/**
 * Obtiene el analytics global
 */
export function getAnalytics(config?: AnalyticsConfig): Analytics {
  if (!globalAnalytics) {
    globalAnalytics = new Analytics(config)
  }
  return globalAnalytics
}

/**
 * Inicializa analytics
 */
export function initAnalytics(config: AnalyticsConfig): Analytics {
  globalAnalytics = new Analytics(config)
  return globalAnalytics
}






