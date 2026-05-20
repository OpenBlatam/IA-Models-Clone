/**
 * Advanced Notification System
 * Enhanced notifications with priorities and actions
 */

import { toast } from 'react-hot-toast'

export type NotificationPriority = 'low' | 'medium' | 'high' | 'critical'
export type NotificationType = 'success' | 'error' | 'warning' | 'info'

interface NotificationOptions {
  priority?: NotificationPriority
  duration?: number
  action?: {
    label: string
    onClick: () => void
  }
  icon?: string
  dismissible?: boolean
}

const PRIORITY_DURATIONS: Record<NotificationPriority, number> = {
  low: 3000,
  medium: 5000,
  high: 8000,
  critical: 0 // Never auto-dismiss
}

export class NotificationSystem {
  private static activeNotifications = new Map<string, string>()

  static show(
    message: string,
    type: NotificationType = 'info',
    options: NotificationOptions = {}
  ): string {
    const {
      priority = 'medium',
      duration,
      action,
      icon,
      dismissible = true
    } = options

    const notificationDuration =
      duration ?? PRIORITY_DURATIONS[priority]

    const notificationId = toast[type](
      (t) => (
        <div className="flex items-center gap-3">
          {icon && <span className="text-xl">{icon}</span>}
          <div className="flex-1">
            <p className="font-medium">{message}</p>
            {action && (
              <button
                onClick={() => {
                  action.onClick()
                  toast.dismiss(t.id)
                }}
                className="mt-2 text-sm text-blue-400 hover:text-blue-300 underline"
              >
                {action.label}
              </button>
            )}
          </div>
          {dismissible && (
            <button
              onClick={() => toast.dismiss(t.id)}
              className="text-gray-400 hover:text-white"
            >
              ×
            </button>
          )}
        </div>
      ),
      {
        duration: notificationDuration,
        id: `notification-${Date.now()}`,
        position: 'top-right',
        style: {
          background: '#1e293b',
          color: '#fff',
          border: '1px solid #334155',
          borderRadius: '8px',
          padding: '16px'
        }
      }
    )

    return notificationId
  }

  static success(message: string, options?: NotificationOptions): string {
    return this.show(message, 'success', {
      ...options,
      icon: options?.icon || '✅'
    })
  }

  static error(message: string, options?: NotificationOptions): string {
    return this.show(message, 'error', {
      ...options,
      icon: options?.icon || '❌',
      priority: 'high'
    })
  }

  static warning(message: string, options?: NotificationOptions): string {
    return this.show(message, 'warning', {
      ...options,
      icon: options?.icon || '⚠️',
      priority: 'medium'
    })
  }

  static info(message: string, options?: NotificationOptions): string {
    return this.show(message, 'info', {
      ...options,
      icon: options?.icon || 'ℹ️',
      priority: 'low'
    })
  }

  static dismiss(id: string): void {
    toast.dismiss(id)
  }

  static dismissAll(): void {
    toast.dismiss()
  }
}

// Export singleton instance
export const notifications = NotificationSystem


