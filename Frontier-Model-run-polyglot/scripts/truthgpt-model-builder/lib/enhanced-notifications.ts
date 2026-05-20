/**
 * Enhanced Notifications
 * Sistema de notificaciones mejorado
 */

export type NotificationType = 'success' | 'error' | 'warning' | 'info' | 'build' | 'complete'

export interface Notification {
  id: string
  type: NotificationType
  title: string
  message: string
  timestamp: number
  read: boolean
  action?: {
    label: string
    callback: () => void
  }
  duration?: number
  priority?: 'low' | 'medium' | 'high'
}

export interface NotificationPreferences {
  enabled: boolean
  sound: boolean
  desktop: boolean
  types: Record<NotificationType, boolean>
  quietHours?: {
    enabled: boolean
    start: number // hour 0-23
    end: number // hour 0-23
  }
}

export class EnhancedNotifications {
  private notifications: Notification[] = []
  private preferences: NotificationPreferences = {
    enabled: true,
    sound: false,
    desktop: false,
    types: {
      success: true,
      error: true,
      warning: true,
      info: true,
      build: true,
      complete: true,
    },
  }
  private storageKey = 'notification-preferences'

  constructor() {
    this.loadPreferences()
  }

  /**
   * Cargar preferencias
   */
  private loadPreferences(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const stored = window.localStorage.getItem(this.storageKey)
        if (stored) {
          this.preferences = { ...this.preferences, ...JSON.parse(stored) }
        }
      }
    } catch (error) {
      console.error('Error loading notification preferences:', error)
    }
  }

  /**
   * Guardar preferencias
   */
  private savePreferences(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(this.storageKey, JSON.stringify(this.preferences))
      }
    } catch (error) {
      console.error('Error saving notification preferences:', error)
    }
  }

  /**
   * Verificar si está en horas silenciosas
   */
  private isQuietHours(): boolean {
    if (!this.preferences.quietHours?.enabled) return false

    const now = new Date()
    const hour = now.getHours()
    const { start, end } = this.preferences.quietHours

    if (start <= end) {
      return hour >= start && hour < end
    } else {
      // Horas que cruzan medianoche
      return hour >= start || hour < end
    }
  }

  /**
   * Crear notificación
   */
  createNotification(
    type: NotificationType,
    title: string,
    message: string,
    options?: {
      action?: { label: string; callback: () => void }
      duration?: number
      priority?: 'low' | 'medium' | 'high'
    }
  ): Notification {
    const notification: Notification = {
      id: `notification-${Date.now()}-${Math.random()}`,
      type,
      title,
      message,
      timestamp: Date.now(),
      read: false,
      action: options?.action,
      duration: options?.duration,
      priority: options?.priority || 'medium',
    }

    // Verificar si está habilitado
    if (!this.preferences.enabled || !this.preferences.types[type]) {
      return notification
    }

    // Verificar horas silenciosas
    if (this.isQuietHours() && options?.priority !== 'high') {
      return notification
    }

    this.notifications.unshift(notification)
    this.notifications = this.notifications.slice(0, 100) // Limitar a 100

    // Mostrar notificación del sistema
    if (this.preferences.desktop && 'Notification' in window) {
      this.showDesktopNotification(title, message, type)
    }

    // Reproducir sonido
    if (this.preferences.sound) {
      this.playSound(type)
    }

    return notification
  }

  /**
   * Mostrar notificación del sistema
   */
  private showDesktopNotification(title: string, message: string, type: NotificationType): void {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(title, {
        body: message,
        icon: this.getIconForType(type),
        badge: '/icon-192x192.png',
        tag: type,
      })
    }
  }

  /**
   * Obtener icono por tipo
   */
  private getIconForType(type: NotificationType): string {
    const icons: Record<NotificationType, string> = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: 'ℹ️',
      build: '🚀',
      complete: '🎉',
    }
    return icons[type] || 'ℹ️'
  }

  /**
   * Reproducir sonido
   */
  private playSound(type: NotificationType): void {
    // En producción, usar archivos de audio reales
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()

    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)

    // Diferentes frecuencias por tipo
    const frequencies: Record<NotificationType, number> = {
      success: 800,
      error: 400,
      warning: 600,
      info: 500,
      build: 700,
      complete: 1000,
    }

    oscillator.frequency.value = frequencies[type] || 500
    oscillator.type = 'sine'
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime)
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1)

    oscillator.start(audioContext.currentTime)
    oscillator.stop(audioContext.currentTime + 0.1)
  }

  /**
   * Solicitar permiso para notificaciones
   */
  async requestPermission(): Promise<boolean> {
    if (!('Notification' in window)) {
      return false
    }

    if (Notification.permission === 'granted') {
      return true
    }

    if (Notification.permission !== 'denied') {
      const permission = await Notification.requestPermission()
      return permission === 'granted'
    }

    return false
  }

  /**
   * Obtener notificaciones
   */
  getNotifications(limit?: number): Notification[] {
    const notifications = this.notifications.slice(0, limit)
    return notifications
  }

  /**
   * Obtener notificaciones no leídas
   */
  getUnreadNotifications(): Notification[] {
    return this.notifications.filter(n => !n.read)
  }

  /**
   * Marcar como leída
   */
  markAsRead(id: string): void {
    const notification = this.notifications.find(n => n.id === id)
    if (notification) {
      notification.read = true
    }
  }

  /**
   * Marcar todas como leídas
   */
  markAllAsRead(): void {
    this.notifications.forEach(n => n.read = true)
  }

  /**
   * Eliminar notificación
   */
  removeNotification(id: string): void {
    this.notifications = this.notifications.filter(n => n.id !== id)
  }

  /**
   * Limpiar todas las notificaciones
   */
  clear(): void {
    this.notifications = []
  }

  /**
   * Obtener preferencias
   */
  getPreferences(): NotificationPreferences {
    return { ...this.preferences }
  }

  /**
   * Actualizar preferencias
   */
  updatePreferences(updates: Partial<NotificationPreferences>): void {
    this.preferences = { ...this.preferences, ...updates }
    this.savePreferences()
  }

  /**
   * Obtener conteo de no leídas
   */
  getUnreadCount(): number {
    return this.notifications.filter(n => !n.read).length
  }
}

// Singleton instance
let enhancedNotificationsInstance: EnhancedNotifications | null = null

export function getEnhancedNotifications(): EnhancedNotifications {
  if (!enhancedNotificationsInstance) {
    enhancedNotificationsInstance = new EnhancedNotifications()
  }
  return enhancedNotificationsInstance
}

export default EnhancedNotifications










