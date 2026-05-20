import { BaseApiClient } from './base-client';
import type { Notification, Reminder } from '@/types';

export class NotificationsApi extends BaseApiClient {
  async getNotifications(userId: string): Promise<Notification[]> {
    const response = await this.client.get<Notification[]>(
      this.getUrl(`/notifications/${userId}`)
    );
    return response.data;
  }

  async markNotificationRead(notificationId: string): Promise<void> {
    await this.client.post(
      this.getUrl(`/notifications/${notificationId}/read`)
    );
  }

  async getReminders(userId: string): Promise<Reminder[]> {
    const response = await this.client.get<Reminder[]>(
      this.getUrl(`/reminders/${userId}`)
    );
    return response.data;
  }
}

export const notificationsApi = new NotificationsApi();

