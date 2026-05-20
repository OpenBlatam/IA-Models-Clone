import { BaseApiClient } from './base-client';
import type {
  AssessmentRequest,
  AssessmentResponse,
} from '@/types';

export class AssessmentApi extends BaseApiClient {
  async assess(data: AssessmentRequest): Promise<AssessmentResponse> {
    const response = await this.client.post<AssessmentResponse>(
      this.getUrl('/assess'),
      data
    );
    return response.data;
  }
}

export const assessmentApi = new AssessmentApi();

