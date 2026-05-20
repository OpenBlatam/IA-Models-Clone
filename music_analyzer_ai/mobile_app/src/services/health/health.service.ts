import { apiClient } from '../api-client';
import { API_ENDPOINTS } from '../../constants/api';
import type { HealthCheck } from '../../types/api';

export class HealthService {
  async healthCheck(): Promise<HealthCheck> {
    return apiClient.get<HealthCheck>(API_ENDPOINTS.HEALTH);
  }
}

export const healthService = new HealthService();

