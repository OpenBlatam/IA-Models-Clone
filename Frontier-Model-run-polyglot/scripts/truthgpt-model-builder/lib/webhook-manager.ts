/**
 * Webhook Manager
 * Sistema de webhooks para integraciones externas
 */

export interface WebhookConfig {
  url: string
  events: WebhookEvent[]
  secret?: string
  enabled: boolean
  retries?: number
  timeout?: number
}

export type WebhookEvent = 
  | 'model.completed'
  | 'model.failed'
  | 'model.started'
  | 'batch.completed'
  | 'queue.empty'
  | 'error.occurred'

export interface WebhookPayload {
  event: WebhookEvent
  timestamp: number
  data: Record<string, any>
  modelId?: string
  modelName?: string
}

export class WebhookManager {
  private webhooks: Map<string, WebhookConfig> = new Map()

  /**
   * Registrar webhook
   */
  registerWebhook(id: string, config: WebhookConfig): void {
    this.webhooks.set(id, { ...config })
  }

  /**
   * Eliminar webhook
   */
  unregisterWebhook(id: string): void {
    this.webhooks.delete(id)
  }

  /**
   * Obtener webhook
   */
  getWebhook(id: string): WebhookConfig | undefined {
    return this.webhooks.get(id)
  }

  /**
   * Obtener todos los webhooks
   */
  getAllWebhooks(): Array<{ id: string; config: WebhookConfig }> {
    return Array.from(this.webhooks.entries()).map(([id, config]) => ({ id, config }))
  }

  /**
   * Disparar webhook
   */
  async triggerWebhook(
    event: WebhookEvent,
    payload: Omit<WebhookPayload, 'event' | 'timestamp'>
  ): Promise<void> {
    const fullPayload: WebhookPayload = {
      event,
      timestamp: Date.now(),
      ...payload,
    }

    const promises = Array.from(this.webhooks.entries())
      .filter(([_, config]) => config.enabled && config.events.includes(event))
      .map(async ([id, config]) => {
        try {
          await this.sendWebhook(id, config, fullPayload)
        } catch (error) {
          console.error(`Error sending webhook ${id}:`, error)
        }
      })

    await Promise.all(promises)
  }

  /**
   * Enviar webhook individual
   */
  private async sendWebhook(
    id: string,
    config: WebhookConfig,
    payload: WebhookPayload
  ): Promise<void> {
    const retries = config.retries || 3
    const timeout = config.timeout || 5000

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeout)

        const headers: Record<string, string> = {
          'Content-Type': 'application/json',
        }

        if (config.secret) {
          headers['X-Webhook-Secret'] = config.secret
        }

        const response = await fetch(config.url, {
          method: 'POST',
          headers,
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        clearTimeout(timeoutId)

        if (response.ok) {
          return // Éxito
        }

        if (response.status >= 400 && response.status < 500) {
          // Error del cliente, no reintentar
          throw new Error(`Client error: ${response.status}`)
        }

        // Error del servidor, reintentar
        if (attempt < retries) {
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000))
          continue
        }

        throw new Error(`Server error: ${response.status}`)
      } catch (error) {
        if (attempt === retries) {
          throw error
        }
        // Esperar antes de reintentar
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000))
      }
    }
  }

  /**
   * Habilitar/deshabilitar webhook
   */
  setWebhookEnabled(id: string, enabled: boolean): void {
    const config = this.webhooks.get(id)
    if (config) {
      config.enabled = enabled
    }
  }

  /**
   * Limpiar todos los webhooks
   */
  clear(): void {
    this.webhooks.clear()
  }
}

// Singleton instance
let webhookManagerInstance: WebhookManager | null = null

export function getWebhookManager(): WebhookManager {
  if (!webhookManagerInstance) {
    webhookManagerInstance = new WebhookManager()
  }
  return webhookManagerInstance
}

export default WebhookManager










