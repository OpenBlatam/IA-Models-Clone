import { apiClient } from '@/lib/api/client';
import { API_CONFIG } from '@/config/api.config';
import type { InspectionResult, BatchInspectionRequest } from './types';

export const inspectionApi = {
  async start(): Promise<boolean> {
    const response = await apiClient.post(API_CONFIG.ENDPOINTS.INSPECTION.START);
    return response.data.success;
  },

  async stop(): Promise<void> {
    await apiClient.post(API_CONFIG.ENDPOINTS.INSPECTION.STOP);
  },

  async inspectFrame(image?: string): Promise<InspectionResult> {
    const response = await apiClient.post(API_CONFIG.ENDPOINTS.INSPECTION.INSPECT, {
      image,
    });
    return response.data;
  },

  async inspectBatch(request: BatchInspectionRequest): Promise<InspectionResult[]> {
    const response = await apiClient.post(API_CONFIG.ENDPOINTS.INSPECTION.BATCH, request);
    return response.data;
  },
};

