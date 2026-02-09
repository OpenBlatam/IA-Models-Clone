import { apiClient } from '@/lib/api/client';
import { API_CONFIG } from '@/config/api.config';
import type { ReportRequest } from './types';

export const reportsApi = {
  async generate(request: ReportRequest): Promise<Blob> {
    const response = await apiClient.post(API_CONFIG.ENDPOINTS.REPORTS.GENERATE, request, {
      responseType: 'blob',
    });
    return response.data;
  },
};

