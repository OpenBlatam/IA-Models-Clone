import { BaseApiClient } from './base-client';

export class HealthApi extends BaseApiClient {
  async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    const response = await this.client.get(this.getUrl('/health'));
    return response.data;
  }
}

export const healthApi = new HealthApi();

