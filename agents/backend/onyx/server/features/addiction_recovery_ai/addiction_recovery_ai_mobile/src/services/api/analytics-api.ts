import { BaseApiClient } from './base-client';
import type { AnalyticsResponse } from '@/types';

export class AnalyticsApi extends BaseApiClient {
  async getAnalytics(userId: string): Promise<AnalyticsResponse> {
    const response = await this.client.get<AnalyticsResponse>(
      this.getUrl(`/analytics/${userId}`)
    );
    return response.data;
  }

  async getAdvancedAnalytics(userId: string): Promise<AnalyticsResponse> {
    const response = await this.client.get<AnalyticsResponse>(
      this.getUrl(`/analytics/advanced/${userId}`)
    );
    return response.data;
  }

  async getInsights(userId: string): Promise<string[]> {
    const response = await this.client.get<string[]>(
      this.getUrl(`/insights/${userId}`)
    );
    return response.data;
  }
}

export const analyticsApi = new AnalyticsApi();

