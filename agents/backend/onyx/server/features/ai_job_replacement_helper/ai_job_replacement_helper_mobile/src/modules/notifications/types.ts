import type { Notification } from '@/types';

export interface NotificationFilters {
  unreadOnly?: boolean;
  type?: string;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  limit?: number;
}

export interface NotificationPreferences {
  email: boolean;
  push: boolean;
  inApp: boolean;
  quietHours: {
    enabled: boolean;
    start: string;
    end: string;
  };
}


