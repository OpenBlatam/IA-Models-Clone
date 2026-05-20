/**
 * Platforms API
 * Handles all platform-related API operations
 */

import { apiGet, apiPost, apiDelete } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { Platform, PlatformConnect } from '@/types';

/**
 * Get all platforms and their connection status
 * @returns Array of platforms
 */
export const getAllPlatforms = async (): Promise<Platform[]> => {
  return apiGet<Platform[]>(API_ENDPOINTS.PLATFORMS);
};

/**
 * Connect a platform
 * @param connection - Platform connection data
 * @returns Connected platform
 */
export const connectPlatform = async (connection: PlatformConnect): Promise<Platform> => {
  return apiPost<Platform>(`${API_ENDPOINTS.PLATFORMS}/connect`, connection);
};

/**
 * Disconnect a platform
 * @param platform - Platform name
 * @returns Deletion result
 */
export const disconnectPlatform = async (platform: string): Promise<void> => {
  return apiDelete<void>(`${API_ENDPOINTS.PLATFORMS}/${platform}`);
};

/**
 * Get platform connection status
 * @param platform - Platform name
 * @returns Platform status
 */
export const getPlatformStatus = async (platform: string): Promise<Platform> => {
  return apiGet<Platform>(`${API_ENDPOINTS.PLATFORMS}/${platform}`);
};


