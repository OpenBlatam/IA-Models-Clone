import { apiClient } from '@/lib/api/client';
import { API_CONFIG } from '@/config/api.config';
import type { Alert, AlertStatistics } from './types';

export const alertsApi = {
  async getRecent(level?: string, limit = 50): Promise<Alert[]> {
    const params = new URLSearchParams();
    if (level) params.append('level', level);
    params.append('limit', limit.toString());
    const response = await apiClient.get(
      `${API_CONFIG.ENDPOINTS.ALERTS.RECENT}?${params.toString()}`
    );
    return response.data;
  },

  async getStatistics(): Promise<AlertStatistics> {
    const response = await apiClient.get(API_CONFIG.ENDPOINTS.ALERTS.STATISTICS);
    return response.data;
  },
};

