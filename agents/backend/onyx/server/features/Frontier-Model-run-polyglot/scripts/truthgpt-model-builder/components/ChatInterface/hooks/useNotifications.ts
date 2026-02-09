/**
 * Custom hook for notifications
 * Handles smart notifications, push notifications, and notification rules
 */

import { useState, useCallback, useEffect } from 'react'

export interface NotificationRule {
  keywords: string[]
  enabled: boolean
  sound?: boolean
  desktop?: boolean
}

export interface NotificationState {
  smartNotifications: boolean
  notificationRules: Map<string, NotificationRule>
  pushNotifications: boolean
  notificationPermission: NotificationPermission
  notificationSettings: {
    sound: boolean
    desktop: boolean
    badge: boolean
  }
  messageNotifications: Map<string, { type: string, timestamp: number }>
}

export interface NotificationActions {
  setSmartNotifications: (enabled: boolean) => void
  addNotificationRule: (name: string, rule: NotificationRule) => void
  removeNotificationRule: (name: string) => void
  updateNotificationRule: (name: string, updates: Partial<NotificationRule>) => void
  setPushNotifications: (enabled: boolean) => void
  requestNotificationPermission: () => Promise<boolean>
  sendNotification: (title: string, options?: NotificationOptions) => void
  updateNotificationSettings: (settings: Partial<NotificationState['notificationSettings']>) => void
  addMessageNotification: (messageId: string, type: string) => void
  clearMessageNotifications: () => void
  checkAndNotify: (message: { id: string, content: string, role: string }) => void
}

const STORAGE_KEY = 'chat-notification-settings'

export function useNotifications(): NotificationState & NotificationActions {
  const [smartNotifications, setSmartNotifications] = useState(true)
  const [notificationRules, setNotificationRules] = useState<Map<string, NotificationRule>>(new Map())
  const [pushNotifications, setPushNotifications] = useState(false)
  const [notificationPermission, setNotificationPermission] = useState<NotificationPermission>('default')
  const [notificationSettings, setNotificationSettings] = useState({
    sound: true,
    desktop: true,
    badge: true,
  })
  const [messageNotifications, setMessageNotifications] = useState<Map<string, { type: string, timestamp: number }>>(new Map())

  // Check notification permission on mount
  useEffect(() => {
    if ('Notification' in window) {
      setNotificationPermission(Notification.permission)
    }
  }, [])

  // Load from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        setSmartNotifications(parsed.smartNotifications !== false)
        setNotificationRules(new Map(parsed.notificationRules || []))
        setPushNotifications(parsed.pushNotifications || false)
        setNotificationSettings(parsed.notificationSettings || {
          sound: true,
          desktop: true,
          badge: true,
        })
      }
    } catch (error) {
      console.error('Error loading notification settings:', error)
    }
  }, [])

  // Save to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        smartNotifications,
        notificationRules: Array.from(notificationRules.entries()),
        pushNotifications,
        notificationSettings,
      }))
    } catch (error) {
      console.error('Error saving notification settings:', error)
    }
  }, [smartNotifications, notificationRules, pushNotifications, notificationSettings])

  const addNotificationRule = useCallback((name: string, rule: NotificationRule) => {
    setNotificationRules(prev => {
      const next = new Map(prev)
      next.set(name, rule)
      return next
    })
  }, [])

  const removeNotificationRule = useCallback((name: string) => {
    setNotificationRules(prev => {
      const next = new Map(prev)
      next.delete(name)
      return next
    })
  }, [])

  const updateNotificationRule = useCallback((name: string, updates: Partial<NotificationRule>) => {
    setNotificationRules(prev => {
      const next = new Map(prev)
      const existing = next.get(name)
      if (existing) {
        next.set(name, { ...existing, ...updates })
      }
      return next
    })
  }, [])

  const requestNotificationPermission = useCallback(async (): Promise<boolean> => {
    if (!('Notification' in window)) {
      return false
    }

    if (Notification.permission === 'granted') {
      setNotificationPermission('granted')
      return true
    }

    if (Notification.permission !== 'denied') {
      const permission = await Notification.requestPermission()
      setNotificationPermission(permission)
      return permission === 'granted'
    }

    return false
  }, [])

  const sendNotification = useCallback((title: string, options: NotificationOptions = {}) => {
    if (!pushNotifications || notificationPermission !== 'granted') {
      return
    }

    if (!notificationSettings.desktop) {
      return
    }

    try {
      const notification = new Notification(title, {
        icon: '/icon.png',
        badge: '/badge.png',
        ...options,
      })

      if (notificationSettings.sound && options.silent !== true) {
        // Play notification sound
        const audio = new Audio('/notification.mp3')
        audio.play().catch(() => {
          // Ignore errors
        })
      }

      notification.onclick = () => {
        window.focus()
        notification.close()
      }

      // Auto-close after 5 seconds
      setTimeout(() => {
        notification.close()
      }, 5000)
    } catch (error) {
      console.error('Error sending notification:', error)
    }
  }, [pushNotifications, notificationPermission, notificationSettings])

  const updateNotificationSettings = useCallback((settings: Partial<NotificationState['notificationSettings']>) => {
    setNotificationSettings(prev => ({
      ...prev,
      ...settings,
    }))
  }, [])

  const addMessageNotification = useCallback((messageId: string, type: string) => {
    setMessageNotifications(prev => {
      const next = new Map(prev)
      next.set(messageId, {
        type,
        timestamp: Date.now(),
      })
      return next
    })
  }, [])

  const clearMessageNotifications = useCallback(() => {
    setMessageNotifications(new Map())
  }, [])

  const checkAndNotify = useCallback((message: { id: string, content: string, role: string }) => {
    if (!smartNotifications) return

    // Check notification rules
    for (const [ruleName, rule] of notificationRules.entries()) {
      if (!rule.enabled) continue

      const shouldNotify = rule.keywords.some(keyword =>
        message.content.toLowerCase().includes(keyword.toLowerCase())
      )

      if (shouldNotify) {
        sendNotification(
          `New message matches rule: ${ruleName}`,
          {
            body: message.content.substring(0, 100),
            tag: message.id,
          }
        )
        addMessageNotification(message.id, ruleName)
        break // Only notify once per message
      }
    }

    // Smart notification for assistant messages
    if (message.role === 'assistant' && smartNotifications) {
      sendNotification(
        'New response',
        {
          body: message.content.substring(0, 100),
          tag: message.id,
        }
      )
    }
  }, [smartNotifications, notificationRules, sendNotification, addMessageNotification])

  return {
    // State
    smartNotifications,
    notificationRules,
    pushNotifications,
    notificationPermission,
    notificationSettings,
    messageNotifications,
    // Actions
    setSmartNotifications,
    addNotificationRule,
    removeNotificationRule,
    updateNotificationRule,
    setPushNotifications,
    requestNotificationPermission,
    sendNotification,
    updateNotificationSettings,
    addMessageNotification,
    clearMessageNotifications,
    checkAndNotify,
  }
}




