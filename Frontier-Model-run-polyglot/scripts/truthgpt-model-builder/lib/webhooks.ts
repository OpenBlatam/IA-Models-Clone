/**
 * Webhook system for model events
 */

import { logger } from './logger'
import { ModelStatus } from './truthgpt-service'

export interface WebhookEvent {
  type: 'model.created' | 'model.completed' | 'model.failed' | 'model.updated'
  modelId: string
  timestamp: Date
  data: any
}

export interface WebhookConfig {
  url: string
  secret?: string
  events: WebhookEvent['type'][]
  timeout?: number
}

class WebhookManager {
  private webhooks: Map<string, WebhookConfig> = new Map()

  /**
   * Register a webhook
   */
  register(id: string, config: WebhookConfig): void {
    this.webhooks.set(id, config)
    logger.info('Webhook registered', { id, url: config.url, events: config.events })
  }

  /**
   * Unregister a webhook
   */
  unregister(id: string): void {
    const deleted = this.webhooks.delete(id)
    if (deleted) {
      logger.info('Webhook unregistered', { id })
    }
    return deleted
  }

  /**
   * Trigger webhooks for an event
   */
  async trigger(event: WebhookEvent): Promise<void> {
    const matchingWebhooks = Array.from(this.webhooks.entries()).filter(
      ([, config]) => config.events.includes(event.type)
    )

    if (matchingWebhooks.length === 0) {
      logger.debug('No webhooks registered for event', { type: event.type })
      return
    }

    logger.info('Triggering webhooks', {
      eventType: event.type,
      count: matchingWebhooks.length,
    })

    // Trigger all matching webhooks in parallel
    await Promise.allSettled(
      matchingWebhooks.map(([id, config]) => this.sendWebhook(id, config, event))
    )
  }

  /**
   * Send webhook request
   */
  private async sendWebhook(
    id: string,
    config: WebhookConfig,
    event: WebhookEvent
  ): Promise<void> {
    try {
      const controller = new AbortController()
      const timeout = config.timeout || 5000

      const timeoutId = setTimeout(() => controller.abort(), timeout)

      const payload = {
        event: event.type,
        modelId: event.modelId,
        timestamp: event.timestamp.toISOString(),
        data: event.data,
      }

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'X-Webhook-Event': event.type,
        'X-Webhook-Timestamp': event.timestamp.toISOString(),
      }

      if (config.secret) {
        // Simple HMAC signature (in production, use crypto.createHmac)
        headers['X-Webhook-Signature'] = `sha256=${config.secret}`
      }

      const response = await fetch(config.url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`Webhook failed with status ${response.status}`)
      }

      logger.debug('Webhook sent successfully', { id, url: config.url, eventType: event.type })
    } catch (error) {
      logger.error(
        'Error sending webhook',
        error instanceof Error ? error : new Error(String(error)),
        { id, url: config.url, eventType: event.type }
      )
      throw error
    }
  }

  /**
   * Get all registered webhooks
   */
  list(): Array<{ id: string; config: WebhookConfig }> {
    return Array.from(this.webhooks.entries()).map(([id, config]) => ({
      id,
      config,
    }))
  }

  /**
   * Clear all webhooks
   */
  clear(): void {
    this.webhooks.clear()
    logger.info('All webhooks cleared')
  }
}

export const webhookManager = new WebhookManager()


