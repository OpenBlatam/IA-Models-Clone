/**
 * API client for social media connections endpoints
 */

import { apiClient } from './client';
import type {
  SocialMediaConnectRequest,
  SocialMediaConnectionResponse,
  SocialMediaPlatform,
} from '../types';

export const connectionsApi = {
  /**
   * Get all connections
   */
  getAll: async (): Promise<SocialMediaConnectionResponse[]> => {
    const response = await apiClient.get<SocialMediaConnectionResponse[]>('/connections');
    return response.data;
  },

  /**
   * Connect social media platform
   */
  connect: async (data: SocialMediaConnectRequest): Promise<SocialMediaConnectionResponse> => {
    const response = await apiClient.post<SocialMediaConnectionResponse>('/connect', data);
    return response.data;
  },

  /**
   * Disconnect social media platform
   */
  disconnect: async (platform: SocialMediaPlatform): Promise<void> => {
    await apiClient.delete(`/connect/${platform}`);
  },
};




