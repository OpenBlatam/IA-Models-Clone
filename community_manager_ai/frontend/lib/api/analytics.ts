/**
 * Analytics API
 * Handles all analytics-related API operations
 */

import { apiGet } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { Analytics, Post } from '@/types';

/**
 * Get platform analytics
 * @param platform - Platform name
 * @param days - Number of days to analyze (default: 7)
 * @returns Platform analytics data
 */
export const getPlatformAnalytics = async (
  platform: string,
  days: number = 7
): Promise<Analytics> => {
  return apiGet<Analytics>(`${API_ENDPOINTS.ANALYTICS}/platform/${platform}`, {
    params: { days },
  });
};

/**
 * Get post analytics
 * @param postId - The post ID
 * @param platform - Optional platform filter
 * @returns Post analytics data
 */
export const getPostAnalytics = async (
  postId: string,
  platform?: string
): Promise<Analytics> => {
  return apiGet<Analytics>(`${API_ENDPOINTS.ANALYTICS}/post/${postId}`, {
    params: platform ? { platform } : undefined,
  });
};

/**
 * Get best performing posts
 * @param platform - Optional platform filter
 * @param limit - Maximum number of results (default: 10)
 * @returns Array of best performing posts
 */
export const getBestPerformingPosts = async (
  platform?: string,
  limit: number = 10
): Promise<Post[]> => {
  return apiGet<Post[]>(`${API_ENDPOINTS.ANALYTICS}/best-performing`, {
    params: { platform, limit },
  });
};

/**
 * Get platform trends
 * @param platform - Platform name
 * @param days - Number of days to analyze (default: 30)
 * @returns Trends data
 */
export const getPlatformTrends = async (
  platform: string,
  days: number = 30
): Promise<Array<{ date: string; engagement: number }>> => {
  return apiGet<Array<{ date: string; engagement: number }>>(
    `${API_ENDPOINTS.ANALYTICS}/trends/${platform}`,
    {
      params: { days },
    }
  );
};

