import { apiService } from '@/services/api';
import type { NotificationFilters } from '../types';
import type { Notification } from '@/types';

export class NotificationService {
  async getNotifications(userId: string, filters: NotificationFilters = {}): Promise<Notification[]> {
    const response = await apiService.getNotifications(
      userId,
      filters.unreadOnly ?? false,
      filters.limit ?? 20
    );
    if (!response.data) {
      throw new Error(response.error || 'Failed to get notifications');
    }
    return response.data;
  }

  async getUnreadCount(userId: string): Promise<number> {
    const response = await apiService.getUnreadCount(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get unread count');
    }
    return response.data;
  }

  async markAsRead(userId: string, notificationId: string): Promise<void> {
    const response = await apiService.markNotificationRead(userId, notificationId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to mark as read');
    }
  }

  async markAllAsRead(userId: string): Promise<void> {
    const response = await apiService.markAllNotificationsRead(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to mark all as read');
    }
  }
}

export const notificationService = new NotificationService();


