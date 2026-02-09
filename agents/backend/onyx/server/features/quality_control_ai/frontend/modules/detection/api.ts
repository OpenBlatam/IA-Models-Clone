import { apiClient } from '@/lib/api/client';
import { API_CONFIG } from '@/config/api.config';
import type { DetectionSettings } from './types';

export const detectionApi = {
  async updateSettings(settings: DetectionSettings): Promise<void> {
    await apiClient.put(API_CONFIG.ENDPOINTS.DETECTION.SETTINGS, settings);
  },
};

