import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { videoService } from '@/services/video-service';

export function useSubmitFeedback() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      feedback,
    }: {
      videoId: string;
      feedback: {
        rating: number;
        comment?: string;
        tags?: string[];
      };
    }) => videoService.submitFeedback(videoId, feedback),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['video', variables.videoId, 'feedback'] });
    },
  });
}

export function useVideoFeedback(videoId: string, enabled = true) {
  return useQuery({
    queryKey: ['video', videoId, 'feedback'],
    queryFn: () => videoService.getVideoFeedback(videoId),
    enabled: enabled && !!videoId,
    staleTime: 60000, // 1 minute
  });
}


