/**
 * Event Bus para comunicación desacoplada entre módulos
 * Implementa el patrón Observer/Publisher-Subscriber
 */

export type EventHandler<T = any> = (data: T) => void | Promise<void>

export interface EventSubscription {
  unsubscribe: () => void
}

export class EventBus {
  private handlers = new Map<string, Set<EventHandler>>()
  private onceHandlers = new Map<string, Set<EventHandler>>()

  /**
   * Suscribe un handler a un evento
   */
  on<T = any>(event: string, handler: EventHandler<T>): EventSubscription {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set())
    }
    this.handlers.get(event)!.add(handler)

    return {
      unsubscribe: () => {
        this.handlers.get(event)?.delete(handler)
      }
    }
  }

  /**
   * Suscribe un handler que se ejecuta solo una vez
   */
  once<T = any>(event: string, handler: EventHandler<T>): EventSubscription {
    if (!this.onceHandlers.has(event)) {
      this.onceHandlers.set(event, new Set())
    }
    this.onceHandlers.get(event)!.add(handler)

    return {
      unsubscribe: () => {
        this.onceHandlers.get(event)?.delete(handler)
      }
    }
  }

  /**
   * Desuscribe un handler de un evento
   */
  off<T = any>(event: string, handler: EventHandler<T>): void {
    this.handlers.get(event)?.delete(handler)
    this.onceHandlers.get(event)?.delete(handler)
  }

  /**
   * Emite un evento
   */
  async emit<T = any>(event: string, data?: T): Promise<void> {
    // Ejecutar handlers regulares
    const handlers = this.handlers.get(event)
    if (handlers) {
      await Promise.all(
        Array.from(handlers).map(handler => {
          try {
            return handler(data)
          } catch (error) {
            console.error(`Error en handler de evento ${event}:`, error)
            return Promise.resolve()
          }
        })
      )
    }

    // Ejecutar handlers de una vez y eliminarlos
    const onceHandlers = this.onceHandlers.get(event)
    if (onceHandlers) {
      await Promise.all(
        Array.from(onceHandlers).map(handler => {
          try {
            return handler(data)
          } catch (error) {
            console.error(`Error en handler de evento ${event}:`, error)
            return Promise.resolve()
          }
        })
      )
      this.onceHandlers.delete(event)
    }
  }

  /**
   * Limpia todos los handlers de un evento
   */
  clear(event: string): void {
    this.handlers.delete(event)
    this.onceHandlers.delete(event)
  }

  /**
   * Limpia todos los handlers
   */
  clearAll(): void {
    this.handlers.clear()
    this.onceHandlers.clear()
  }

  /**
   * Obtiene el número de handlers para un evento
   */
  listenerCount(event: string): number {
    const regular = this.handlers.get(event)?.size || 0
    const once = this.onceHandlers.get(event)?.size || 0
    return regular + once
  }
}

/**
 * Instancia singleton del EventBus
 */
export const eventBus = new EventBus()

/**
 * Tipos de eventos predefinidos
 */
export enum MessageEvents {
  ATTACHMENT_ADDED = 'message:attachment:added',
  ATTACHMENT_REMOVED = 'message:attachment:removed',
  LINK_ADDED = 'message:link:added',
  LINK_REMOVED = 'message:link:removed',
  NOTIFICATION_ADDED = 'message:notification:added',
  BOOKMARK_ADDED = 'message:bookmark:added',
  BOOKMARK_REMOVED = 'message:bookmark:removed',
  HIGHLIGHT_ADDED = 'message:highlight:added',
  HIGHLIGHT_REMOVED = 'message:highlight:removed',
  ANNOTATION_ADDED = 'message:annotation:added',
  ANNOTATION_REMOVED = 'message:annotation:removed',
  POLL_CREATED = 'message:poll:created',
  POLL_VOTED = 'message:poll:voted',
  WORKFLOW_CREATED = 'message:workflow:created',
  TASK_CREATED = 'message:task:created',
  REMINDER_SET = 'message:reminder:set',
  EVENT_SCHEDULED = 'message:event:scheduled'
}



