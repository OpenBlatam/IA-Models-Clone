import { BaseApiClient } from './base-client';
import type { DashboardResponse } from '@/types';

export class DashboardApi extends BaseApiClient {
  async getDashboard(userId: string): Promise<DashboardResponse> {
    const response = await this.client.get<DashboardResponse>(
      this.getUrl(`/dashboard/${userId}`)
    );
    return response.data;
  }
}

export const dashboardApi = new DashboardApi();

