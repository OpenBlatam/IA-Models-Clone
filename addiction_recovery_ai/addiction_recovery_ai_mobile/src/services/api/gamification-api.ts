import { BaseApiClient } from './base-client';
import type { PointsResponse, Achievement } from '@/types';

export class GamificationApi extends BaseApiClient {
  async getPoints(userId: string): Promise<PointsResponse> {
    const response = await this.client.get<PointsResponse>(
      this.getUrl(`/gamification/points/${userId}`)
    );
    return response.data;
  }

  async getGamificationAchievements(userId: string): Promise<Achievement[]> {
    const response = await this.client.get<Achievement[]>(
      this.getUrl(`/gamification/achievements/${userId}`)
    );
    return response.data;
  }
}

export const gamificationApi = new GamificationApi();

