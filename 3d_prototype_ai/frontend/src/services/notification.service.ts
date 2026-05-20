import { apiClient } from '@/lib/api-client'
import type { Notification } from '@/types'

export const notificationService = {
  // Obtener notificaciones
  getNotifications: async (): Promise<Notification[]> => {
    return apiClient.get<Notification[]>('/api/v1/notifications')
  },

  // Marcar como leída
  markAsRead: async (id: string): Promise<void> => {
    await apiClient.post(`/api/v1/notifications/${id}/read`)
  },

  // Marcar todas como leídas
  markAllAsRead: async (): Promise<void> => {
    await apiClient.post('/api/v1/notifications/read-all')
  },

  // Obtener notificaciones push
  getPushNotifications: async (userId: string): Promise<Notification[]> => {
    return apiClient.get<Notification[]>(`/api/v1/notifications/${userId}`)
  },
}



