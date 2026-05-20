/**
 * Servicio específico para notificaciones
 */

import type { MessageNotification } from '../../types/message.types'
import {
  addToMap,
  getFromMap,
  removeFromMap,
  filterMap
} from '../../utils/mapUtils'

export class NotificationService {
  static create(
    notifications: Map<string, MessageNotification>,
    key: string,
    notification: MessageNotification
  ): Map<string, MessageNotification> {
    return addToMap(notifications, key, notification)
  }

  static get(
    notifications: Map<string, MessageNotification>,
    key: string
  ): MessageNotification | undefined {
    return getFromMap(notifications, key)
  }

  /**
   * Marca una notificación como leída
   */
  static markAsRead(
    notifications: Map<string, MessageNotification>,
    key: string
  ): Map<string, MessageNotification> {
    const newMap = new Map(notifications)
    const notification = newMap.get(key)
    if (notification) {
      newMap.set(key, { ...notification, read: true })
    }
    return newMap
  }

  /**
   * Marca una notificación como no leída
   */
  static markAsUnread(
    notifications: Map<string, MessageNotification>,
    key: string
  ): Map<string, MessageNotification> {
    const newMap = new Map(notifications)
    const notification = newMap.get(key)
    if (notification) {
      newMap.set(key, { ...notification, read: false })
    }
    return newMap
  }

  static getUnread(
    notifications: Map<string, MessageNotification>
  ): Map<string, MessageNotification> {
    return filterMap(notifications, (notification) => !notification.read)
  }

  /**
   * Cuenta las notificaciones no leídas
   */
  static countUnread(
    notifications: Map<string, MessageNotification>
  ): number {
    return this.getUnread(notifications).size
  }

  /**
   * Elimina una notificación
   */
  static remove(
    notifications: Map<string, MessageNotification>,
    key: string
  ): Map<string, MessageNotification> {
    const newMap = new Map(notifications)
    newMap.delete(key)
    return newMap
  }

  static clearRead(
    notifications: Map<string, MessageNotification>
  ): Map<string, MessageNotification> {
    return filterMap(notifications, (notification) => !notification.read)
  }
}

