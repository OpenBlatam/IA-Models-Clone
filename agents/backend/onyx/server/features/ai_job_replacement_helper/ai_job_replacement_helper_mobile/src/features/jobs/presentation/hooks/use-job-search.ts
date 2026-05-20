import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuthStore } from '@/features/auth';
import { JobService } from '../../domain/job-service';
import type { JobSearchParams, JobSwipeAction, JobApplication } from '../../domain/job-types';

const jobService = new JobService();

export function useJobSearch(params: JobSearchParams) {
  const { user } = useAuthStore();

  return useQuery({
    queryKey: ['jobs', 'search', user?.id, params],
    queryFn: () => {
      if (!user?.id) throw new Error('User not authenticated');
      return jobService.searchJobs(user.id, params);
    },
    enabled: !!user?.id,
  });
}

export function useJobSwipe() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (action: JobSwipeAction) => {
      if (!user?.id) throw new Error('User not authenticated');
      return jobService.swipeJob(user.id, action);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs', user?.id] });
    },
  });
}

export function useJobApplication() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (application: JobApplication) => {
      if (!user?.id) throw new Error('User not authenticated');
      return jobService.applyToJob(user.id, application);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs', user?.id] });
    },
  });
}

