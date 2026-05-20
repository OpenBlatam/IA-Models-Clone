import { apiClient } from '../api-client';
import type { AlertResponse } from '@/types/api';

export const alertsApi = {
  getAll: async (): Promise<AlertResponse[]> => {
    const response = await apiClient.get<AlertResponse[]>('/alerts');
    return response.data;
  },

  getById: async (alertId: string): Promise<AlertResponse> => {
    const response = await apiClient.get<AlertResponse>(`/alerts/${alertId}`);
    return response.data;
  },

  markAsRead: async (alertId: string): Promise<AlertResponse> => {
    const response = await apiClient.patch<AlertResponse>(`/alerts/${alertId}/read`);
    return response.data;
  },

  delete: async (alertId: string): Promise<void> => {
    await apiClient.delete(`/alerts/${alertId}`);
  },
};




