/**
 * Notification Manager
 * Gestiona notificaciones push del navegador y del sistema
 */

export interface NotificationOptions {
  title: string
  body: string
  icon?: string
  badge?: string
  tag?: string
  requireInteraction?: boolean
  silent?: boolean
  timestamp?: number
  data?: any
}

class NotificationManager {
  private permission: NotificationPermission = 'default'
  private isSupported: boolean = false

  constructor() {
    if (typeof window !== 'undefined') {
      this.isSupported = 'Notification' in window
      this.permission = Notification.permission
    }
  }

  /**
   * Solicitar permiso para notificaciones
   */
  async requestPermission(): Promise<boolean> {
    if (!this.isSupported) {
      console.warn('Notifications not supported in this browser')
      return false
    }

    if (this.permission === 'granted') {
      return true
    }

    if (this.permission === 'denied') {
      console.warn('Notification permission denied')
      return false
    }

    try {
      const permission = await Notification.requestPermission()
      this.permission = permission
      return permission === 'granted'
    } catch (error) {
      console.error('Error requesting notification permission:', error)
      return false
    }
  }

  /**
   * Enviar notificación
   */
  async showNotification(options: NotificationOptions): Promise<void> {
    if (!this.isSupported) {
      console.warn('Notifications not supported')
      return
    }

    if (this.permission !== 'granted') {
      const granted = await this.requestPermission()
      if (!granted) {
        return
      }
    }

    try {
      const notificationOptions: NotificationOptions = {
        icon: options.icon || '/icon-192x192.png',
        badge: options.badge || '/badge-72x72.png',
        tag: options.tag || 'model-builder',
        requireInteraction: options.requireInteraction || false,
        silent: options.silent || false,
        timestamp: options.timestamp || Date.now(),
        data: options.data,
      }

      const notification = new Notification(options.title, notificationOptions)

      notification.onclick = () => {
        window.focus()
        notification.close()
      }

      // Auto-close after 5 seconds unless requireInteraction is true
      if (!notificationOptions.requireInteraction) {
        setTimeout(() => {
          notification.close()
        }, 5000)
      }
    } catch (error) {
      console.error('Error showing notification:', error)
    }
  }

  /**
   * Notificación de modelo completado
   */
  async notifyModelCompleted(modelName: string, duration?: number): Promise<void> {
    await this.showNotification({
      title: '✅ Modelo Completado',
      body: `${modelName} ha sido construido exitosamente${duration ? ` en ${Math.round(duration / 1000)}s` : ''}`,
      tag: `model-completed-${Date.now()}`,
      data: { type: 'model-completed', modelName },
    })
  }

  /**
   * Notificación de modelo fallido
   */
  async notifyModelFailed(modelName: string, error?: string): Promise<void> {
    await this.showNotification({
      title: '❌ Error en Modelo',
      body: `Error al construir ${modelName}${error ? `: ${error}` : ''}`,
      tag: `model-failed-${Date.now()}`,
      requireInteraction: true,
      data: { type: 'model-failed', modelName, error },
    })
  }

  /**
   * Notificación de construcción iniciada
   */
  async notifyBuildStarted(modelName: string): Promise<void> {
    await this.showNotification({
      title: '🚀 Construcción Iniciada',
      body: `Iniciando construcción de ${modelName}`,
      tag: `build-started-${Date.now()}`,
      silent: true,
      data: { type: 'build-started', modelName },
    })
  }

  /**
   * Notificación de cola vacía
   */
  async notifyQueueEmpty(): Promise<void> {
    await this.showNotification({
      title: '📋 Cola Completada',
      body: 'Todos los modelos han sido procesados',
      tag: 'queue-empty',
      data: { type: 'queue-empty' },
    })
  }

  /**
   * Obtener estado de permisos
   */
  getPermissionStatus(): NotificationPermission {
    return this.permission
  }

  /**
   * Verificar si está soportado
   */
  isNotificationSupported(): boolean {
    return this.isSupported
  }
}

// Singleton instance
let notificationManagerInstance: NotificationManager | null = null

export function getNotificationManager(): NotificationManager {
  if (!notificationManagerInstance) {
    notificationManagerInstance = new NotificationManager()
  }
  return notificationManagerInstance
}

export default NotificationManager










