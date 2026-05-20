import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { videoService } from '@/services/video-service';
import type { VideoGenerationRequest } from '@/types/api';

export function useScheduleVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      options,
    }: {
      videoId: string;
      options: {
        scheduled_at: string;
        request: VideoGenerationRequest;
        timezone?: string;
        repeat?: string;
      };
    }) => videoService.scheduleVideo(videoId, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scheduled'] });
    },
  });
}

export function useScheduledVideos(
  videoId?: string,
  status?: string,
  enabled = true
) {
  return useQuery({
    queryKey: ['scheduled', videoId, status],
    queryFn: async () => {
      // This would need to be implemented in the service
      // For now, returning empty array
      return { jobs: [] };
    },
    enabled,
    refetchInterval: 30000, // Poll every 30 seconds
  });
}

export function useCancelScheduledJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      // This would need to be implemented in the service
      return { message: 'Job cancelled' };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scheduled'] });
    },
  });
}


