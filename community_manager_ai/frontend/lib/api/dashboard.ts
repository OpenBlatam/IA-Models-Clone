/**
 * Dashboard API
 * Handles all dashboard-related API operations
 */

import { apiGet } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { DashboardOverview, EngagementSummary, Post } from '@/types';

export interface Activity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

/**
 * Get dashboard overview
 * @param days - Number of days to analyze (default: 7)
 * @returns Dashboard overview data
 */
export const getDashboardOverview = async (days: number = 7): Promise<DashboardOverview> => {
  return apiGet<DashboardOverview>(`${API_ENDPOINTS.DASHBOARD}/overview`, {
    params: { days },
  });
};

/**
 * Get engagement summary
 * @param days - Number of days to analyze (default: 7)
 * @returns Engagement summary data
 */
export const getEngagementSummary = async (days: number = 7): Promise<EngagementSummary> => {
  return apiGet<EngagementSummary>(`${API_ENDPOINTS.DASHBOARD}/engagement`, {
    params: { days },
  });
};

/**
 * Get upcoming posts
 * @param limit - Maximum number of results (default: 10)
 * @returns Array of upcoming posts
 */
export const getUpcomingPosts = async (limit: number = 10): Promise<Post[]> => {
  return apiGet<Post[]>(`${API_ENDPOINTS.DASHBOARD}/upcoming-posts`, {
    params: { limit },
  });
};

/**
 * Get recent activity
 * @param limit - Maximum number of results (default: 20)
 * @returns Array of recent activities
 */
export const getRecentActivity = async (limit: number = 20): Promise<Activity[]> => {
  return apiGet<Activity[]>(`${API_ENDPOINTS.DASHBOARD}/recent-activity`, {
    params: { limit },
  });
};

