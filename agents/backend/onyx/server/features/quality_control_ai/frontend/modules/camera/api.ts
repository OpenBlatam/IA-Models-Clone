import { apiClient } from '@/lib/api/client';
import { API_CONFIG } from '@/config/api.config';
import type { CameraInfo, CameraSettings } from './types';

export const cameraApi = {
  async initialize(): Promise<boolean> {
    const response = await apiClient.post(API_CONFIG.ENDPOINTS.CAMERA.INITIALIZE);
    return response.data.success;
  },

  async getInfo(): Promise<CameraInfo> {
    const response = await apiClient.get(API_CONFIG.ENDPOINTS.CAMERA.INFO);
    return response.data;
  },

  async updateSettings(settings: CameraSettings): Promise<void> {
    await apiClient.put(API_CONFIG.ENDPOINTS.CAMERA.SETTINGS, settings);
  },

  async captureFrame(): Promise<string> {
    const response = await apiClient.get(API_CONFIG.ENDPOINTS.CAMERA.CAPTURE, {
      responseType: 'blob',
    });
    const blob = response.data;
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
  },
};

