import { useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import { queryKeys } from './queryKeys';
import { REFETCH_INTERVALS } from '@/lib/constants';
import type { IdentityProfile, Template, Task, DashboardResponse, MetricsResponse } from '@/types';

export const useIdentities = () => {
  return useQuery(queryKeys.identities.lists(), async () => {
    try {
      return await apiClient.getIdentities();
    } catch {
      return await apiClient.searchIdentities({});
    }
  });
};

export const useIdentity = (id: string) => {
  return useQuery(
    queryKeys.identities.detail(id),
    () => apiClient.getIdentity(id),
    { enabled: !!id }
  );
};

export const useGeneratedContent = (identityId: string, limit = 10) => {
  return useQuery(
    queryKeys.identities.generatedContent(identityId),
    () => apiClient.getGeneratedContent(identityId, limit),
    { enabled: !!identityId }
  );
};

export const useTemplates = () => {
  return useQuery(queryKeys.templates.lists(), () => apiClient.getTemplates());
};

export const useTemplate = (id: string) => {
  return useQuery(
    queryKeys.templates.detail(id),
    () => apiClient.getTemplate(id),
    { enabled: !!id }
  );
};

export const useTasks = () => {
  return useQuery(
    queryKeys.tasks.lists(),
    () => apiClient.getTasks(),
    { refetchInterval: REFETCH_INTERVALS.TASKS }
  );
};

export const useTask = (id: string) => {
  return useQuery(
    queryKeys.tasks.detail(id),
    () => apiClient.getTask(id),
    { enabled: !!id }
  );
};

export const useAlerts = (params?: { unacknowledged_only?: boolean; severity?: string }) => {
  return useQuery(
    [...queryKeys.alerts.lists(), params],
    () => apiClient.getAlerts(params)
  );
};

export const useDashboard = () => {
  return useQuery(
    queryKeys.dashboard.all,
    () => apiClient.getDashboard(),
    { staleTime: 30000, cacheTime: 300000 }
  );
};

export const useMetrics = () => {
  return useQuery(
    queryKeys.metrics.all,
    () => apiClient.getMetrics(),
    {
      refetchInterval: REFETCH_INTERVALS.METRICS,
      staleTime: 10000,
      cacheTime: 60000,
    }
  );
};

export const useAnalyticsStats = () => {
  return useQuery(queryKeys.analytics.stats(), () => apiClient.getAnalyticsStats());
};

export const useAnalyticsTrends = () => {
  return useQuery(queryKeys.analytics.trends(), () => apiClient.getAnalyticsTrends());
};

export const useIdentityAnalytics = (identityId: string) => {
  return useQuery(
    queryKeys.identities.analytics(identityId),
    () => apiClient.getAnalyticsForIdentity(identityId),
    { enabled: !!identityId, staleTime: 30000 }
  );
};

