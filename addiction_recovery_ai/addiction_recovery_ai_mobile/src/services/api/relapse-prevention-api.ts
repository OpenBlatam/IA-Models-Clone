import { BaseApiClient } from './base-client';
import type {
  CheckRelapseRiskRequest,
  RelapseRiskResponse,
  TriggerResponse,
} from '@/types';

export class RelapsePreventionApi extends BaseApiClient {
  async checkRelapseRisk(data: CheckRelapseRiskRequest): Promise<RelapseRiskResponse> {
    const response = await this.client.post<RelapseRiskResponse>(
      this.getUrl('/check-relapse-risk'),
      data
    );
    return response.data;
  }

  async getTriggers(userId: string): Promise<TriggerResponse> {
    const response = await this.client.get<TriggerResponse>(
      this.getUrl(`/triggers/${userId}`)
    );
    return response.data;
  }

  async getCopingStrategies(data: {
    user_id: string;
    trigger?: string;
  }): Promise<string[]> {
    const response = await this.client.post<string[]>(
      this.getUrl('/coping-strategies'),
      data
    );
    return response.data;
  }
}

export const relapsePreventionApi = new RelapsePreventionApi();

