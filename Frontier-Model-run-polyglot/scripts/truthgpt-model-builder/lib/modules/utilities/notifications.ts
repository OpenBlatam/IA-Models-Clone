/**
 * Unified Notification System
 * Consolidates notification-manager, notification-system, and enhanced-notifications
 */

import { toast } from 'react-hot-toast'

export type NotificationType = 'success' | 'error' | 'warning' | 'info' | 'build' | 'complete'
export type NotificationPriority = 'low' | 'medium' | 'high' | 'critical'

export interface NotificationOptions {
  priority?: NotificationPriority
  duration?: number
  action?: {
    label: string
    onClick: () => void
  }
  icon?: string
  dismissible?: boolean
  title?: string
  body?: string
  badge?: string
  tag?: string
  requireInteraction?: boolean
  silent?: boolean
  timestamp?: number
  data?: any
  read?: boolean
}

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
  priority?: NotificationPriority
}

const PRIORITY_DURATIONS: Record<NotificationPriority, number> = {
  low: 3000,
  medium: 5000,
  high: 8000,
  critical: 0
}

class UnifiedNotificationSystem {
  private notifications: Notification[] = []
  private permission: NotificationPermission = 'default'
  private isSupported: boolean = false

  constructor() {
    if (typeof window !== 'undefined') {
      this.isSupported = 'Notification' in window
      this.permission = Notification.permission
    }
  }

  /**
   * Show a notification (toast + browser notification if enabled)
   */
  show(
    message: string,
    type: NotificationType = 'info',
    options: NotificationOptions = {}
  ): string {
    const {
      priority = 'medium',
      duration,
      action,
      icon,
      dismissible = true,
      title
    } = options

    const notificationDuration = duration ?? PRIORITY_DURATIONS[priority]
    const notificationId = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

    const toastOptions: any = {
      duration: notificationDuration,
      id: notificationId,
    }

    if (action) {
      toast[type](
        (t) => (
          <div className="flex items-center gap-3">
            <div className="flex-1">
              {title && <div className="font-semibold">{title}</div>}
              <div>{message}</div>
            </div>
            {action && (
              <button
                onClick={() => {
                  action.onClick()
                  toast.dismiss(t.id)
                }}
                className="px-3 py-1 bg-white/20 rounded hover:bg-white/30 transition"
              >
                {action.label}
              </button>
            )}
          </div>
        ),
        toastOptions
      )
    } else {
      toast[type](message, toastOptions)
    }

    const notification: Notification = {
      id: notificationId,
      type,
      title: title || message,
      message,
      timestamp: Date.now(),
      read: false,
      action: action ? { label: action.label, callback: action.onClick } : undefined,
      duration: notificationDuration,
      priority,
    }

    this.notifications.push(notification)

    if (this.isSupported && this.permission === 'granted' && options.requireInteraction) {
      this.showBrowserNotification(title || message, message, options)
    }

    return notificationId
  }

  /**
   * Show browser notification
   */
  private showBrowserNotification(
    title: string,
    body: string,
    options: NotificationOptions
  ): void {
    if (!this.isSupported || this.permission !== 'granted') {
      return
    }

    try {
      const notification = new Notification(title, {
        body,
        icon: options.icon,
        badge: options.badge,
        tag: options.tag,
        requireInteraction: options.requireInteraction,
        silent: options.silent,
        timestamp: options.timestamp || Date.now(),
        data: options.data,
      })

      notification.onclick = () => {
        window.focus()
        notification.close()
        if (options.action) {
          options.action.onClick()
        }
      }
    } catch (error) {
      console.error('Error showing browser notification:', error)
    }
  }

  /**
   * Request notification permission
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
   * Get all notifications
   */
  getAll(): Notification[] {
    return [...this.notifications]
  }

  /**
   * Mark notification as read
   */
  markAsRead(id: string): void {
    const notification = this.notifications.find(n => n.id === id)
    if (notification) {
      notification.read = true
    }
  }

  /**
   * Clear all notifications
   */
  clear(): void {
    this.notifications = []
    toast.dismiss()
  }

  /**
   * Dismiss notification
   */
  dismiss(id: string): void {
    toast.dismiss(id)
    this.notifications = this.notifications.filter(n => n.id !== id)
  }

  /**
   * Get unread count
   */
  getUnreadCount(): number {
    return this.notifications.filter(n => !n.read).length
  }
}

export const notifications = new UnifiedNotificationSystem()

export default notifications

