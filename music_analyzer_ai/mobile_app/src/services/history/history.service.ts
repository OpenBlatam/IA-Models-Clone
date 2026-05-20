import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type { HistoryResponse, HistoryStats } from '../../types/api';

export class HistoryService {
  async getHistory(limit = 50): Promise<HistoryResponse> {
    if (limit < 1 || limit > 100) {
      throw new Error('Limit must be between 1 and 100');
    }
    return apiClient.get<HistoryResponse>(
      `${API_ENDPOINTS.HISTORY}?limit=${limit}`
    );
  }

  async getHistoryStats(): Promise<HistoryStats> {
    return apiClient.get<HistoryStats>(API_ENDPOINTS.HISTORY_STATS);
  }
}

export const historyService = new HistoryService();

