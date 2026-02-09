import { apiService } from '@/services/api';
import type { DashboardData } from '@/types';

export class DashboardService {
  async getDashboard(userId: string): Promise<DashboardData> {
    const response = await apiService.getDashboard(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get dashboard');
    }
    return response.data;
  }

  async getMetrics(userId: string) {
    const response = await apiService.getDashboardMetrics(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get metrics');
    }
    return response.data;
  }

  async getActivityStats(userId: string, days: number = 30) {
    const response = await apiService.getActivityStats(userId, days);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get activity stats');
    }
    return response.data;
  }
}

export const dashboardService = new DashboardService();


