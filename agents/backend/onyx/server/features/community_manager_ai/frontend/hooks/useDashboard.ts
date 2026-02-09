/**
 * Dashboard Hooks
 * React Query hooks for dashboard-related operations
 */

import { useQuery } from '@tanstack/react-query';
import { dashboardApi } from '@/lib/api';
import { QUERY_KEYS, DEFAULTS } from '@/lib/config/constants';

/**
 * Hook to fetch dashboard overview
 * @param days - Number of days to analyze (default: 7)
 * @returns React Query result with dashboard overview data
 */
export const useDashboardOverview = (days: number = DEFAULTS.DASHBOARD_DAYS) => {
  return useQuery({
    queryKey: QUERY_KEYS.dashboard.overview(days),
    queryFn: () => dashboardApi.getDashboardOverview(days),
  });
};

/**
 * Hook to fetch engagement summary
 * @param days - Number of days to analyze (default: 7)
 * @returns React Query result with engagement summary data
 */
export const useDashboardEngagement = (days: number = DEFAULTS.DASHBOARD_DAYS) => {
  return useQuery({
    queryKey: QUERY_KEYS.dashboard.engagement(days),
    queryFn: () => dashboardApi.getEngagementSummary(days),
  });
};

/**
 * Hook to fetch upcoming posts
 * @param limit - Maximum number of results (default: 10)
 * @returns React Query result with upcoming posts
 */
export const useUpcomingPosts = (limit: number = DEFAULTS.UPCOMING_POSTS_LIMIT) => {
  return useQuery({
    queryKey: QUERY_KEYS.dashboard.upcomingPosts(limit),
    queryFn: () => dashboardApi.getUpcomingPosts(limit),
  });
};

/**
 * Hook to fetch recent activity
 * @param limit - Maximum number of results (default: 20)
 * @returns React Query result with recent activity
 */
export const useRecentActivity = (limit: number = DEFAULTS.RECENT_ACTIVITY_LIMIT) => {
  return useQuery({
    queryKey: QUERY_KEYS.dashboard.recentActivity(limit),
    queryFn: () => dashboardApi.getRecentActivity(limit),
  });
};


