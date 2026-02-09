import { BaseApiClient } from './base-client';
import type {
  CreateRecoveryPlanRequest,
  RecoveryPlanResponse,
} from '@/types';

export class RecoveryPlanApi extends BaseApiClient {
  async createPlan(data: CreateRecoveryPlanRequest): Promise<RecoveryPlanResponse> {
    const response = await this.client.post<RecoveryPlanResponse>(
      this.getUrl('/create-plan'),
      data
    );
    return response.data;
  }

  async getPlan(userId: string): Promise<RecoveryPlanResponse> {
    const response = await this.client.get<RecoveryPlanResponse>(
      this.getUrl(`/plan/${userId}`)
    );
    return response.data;
  }

  async getStrategies(addictionType: string): Promise<string[]> {
    const response = await this.client.get<string[]>(
      this.getUrl(`/strategies/${addictionType}`)
    );
    return response.data;
  }
}

export const recoveryPlanApi = new RecoveryPlanApi();

