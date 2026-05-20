import { BaseApiClient } from './base-client';
import type {
  LogEntryRequest,
  LogEntryResponse,
  ProgressResponse,
  StatsResponse,
  TimelineResponse,
} from '@/types';

export class ProgressApi extends BaseApiClient {
  async logEntry(data: LogEntryRequest): Promise<LogEntryResponse> {
    const response = await this.client.post<LogEntryResponse>(
      this.getUrl('/log-entry'),
      data
    );
    return response.data;
  }

  async getProgress(userId: string): Promise<ProgressResponse> {
    const response = await this.client.get<ProgressResponse>(
      this.getUrl(`/progress/${userId}`)
    );
    return response.data;
  }

  async getStats(userId: string): Promise<StatsResponse> {
    const response = await this.client.get<StatsResponse>(
      this.getUrl(`/stats/${userId}`)
    );
    return response.data;
  }

  async getTimeline(userId: string): Promise<TimelineResponse> {
    const response = await this.client.get<TimelineResponse>(
      this.getUrl(`/timeline/${userId}`)
    );
    return response.data;
  }
}

export const progressApi = new ProgressApi();

