import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { offlineCache } from '../services/offlineCache';
import { Project, ProjectRequest, ApiError, Stats, QueueStatus } from '../types';

export const useProjectsQuery = (params?: {
  status?: string;
  author?: string;
  limit?: number;
  offset?: number;
}) => {
  return useQuery<Project[], ApiError>({
    queryKey: ['projects', params],
    queryFn: async () => {
      try {
        const projects = await apiService.listProjects(params);
        await offlineCache.setProjects(projects);
        return projects;
      } catch (error) {
        const cached = await offlineCache.getProjects();
        if (cached) {
          return cached;
        }
        throw error;
      }
    },
    staleTime: 30000,
    gcTime: 10 * 60 * 1000,
    retry: 2,
  });
};

export const useProjectQuery = (projectId: string | null) => {
  return useQuery<Project, ApiError>({
    queryKey: ['project', projectId],
    queryFn: () => apiService.getProject(projectId!),
    enabled: !!projectId,
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });
};

export const useStatsQuery = () => {
  return useQuery<Stats, ApiError>({
    queryKey: ['stats'],
    queryFn: async () => {
      try {
        const stats = await apiService.getStats();
        await offlineCache.setStats(stats);
        return stats;
      } catch (error) {
        const cached = await offlineCache.getStats();
        if (cached) {
          return cached;
        }
        throw error;
      }
    },
    refetchInterval: 30000,
    staleTime: 30000,
    retry: 2,
  });
};

export const useQueueStatusQuery = () => {
  return useQuery<QueueStatus, ApiError>({
    queryKey: ['queueStatus'],
    queryFn: () => apiService.getQueueStatus(),
    refetchInterval: 10000,
    staleTime: 10000,
    retry: 2,
  });
};

export const useGenerateProjectMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      request,
      asyncGeneration,
    }: {
      request: ProjectRequest;
      asyncGeneration?: boolean;
    }) => apiService.generateProject(request, asyncGeneration),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      queryClient.invalidateQueries({ queryKey: ['queueStatus'] });
    },
  });
};

export const useDeleteProjectMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (projectId: string) => apiService.deleteProject(projectId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      queryClient.invalidateQueries({ queryKey: ['queueStatus'] });
    },
  });
};
