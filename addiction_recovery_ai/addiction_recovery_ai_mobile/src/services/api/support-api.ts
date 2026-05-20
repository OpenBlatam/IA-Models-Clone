import { BaseApiClient } from './base-client';
import type {
  CoachingSessionRequest,
  CoachingSessionResponse,
  MotivationResponse,
  Achievement,
} from '@/types';

export class SupportApi extends BaseApiClient {
  async coachingSession(data: CoachingSessionRequest): Promise<CoachingSessionResponse> {
    const response = await this.client.post<CoachingSessionResponse>(
      this.getUrl('/coaching-session'),
      data
    );
    return response.data;
  }

  async getMotivation(userId: string): Promise<MotivationResponse> {
    const response = await this.client.get<MotivationResponse>(
      this.getUrl(`/motivation/${userId}`)
    );
    return response.data;
  }

  async celebrateMilestone(data: {
    user_id: string;
    milestone: string;
    days_sober: number;
  }): Promise<void> {
    await this.client.post(this.getUrl('/celebrate-milestone'), data);
  }

  async getAchievements(userId: string): Promise<Achievement[]> {
    const response = await this.client.get<Achievement[]>(
      this.getUrl(`/achievements/${userId}`)
    );
    return response.data;
  }
}

export const supportApi = new SupportApi();

