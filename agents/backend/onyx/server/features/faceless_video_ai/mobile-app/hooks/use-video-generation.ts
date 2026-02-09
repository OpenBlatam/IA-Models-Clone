import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { videoService } from '@/services/video-service';
import type { VideoGenerationRequest, VideoGenerationResponse } from '@/types/api';

export function useGenerateVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: VideoGenerationRequest) => videoService.generateVideo(request),
    onSuccess: (data) => {
      // Invalidate and refetch videos list
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      // Set the new video in cache
      queryClient.setQueryData(['video', data.video_id], data);
    },
  });
}

export function useVideoStatus(videoId: string, enabled = true) {
  return useQuery({
    queryKey: ['video', videoId, 'status'],
    queryFn: () => videoService.getVideoStatus(videoId),
    enabled: enabled && !!videoId,
    refetchInterval: (query) => {
      const data = query.state.data as VideoGenerationResponse | undefined;
      // Poll every 2 seconds if video is still processing
      if (
        data?.status === 'pending' ||
        data?.status === 'processing' ||
        data?.status === 'generating_images' ||
        data?.status === 'generating_audio' ||
        data?.status === 'adding_subtitles' ||
        data?.status === 'compositing'
      ) {
        return 2000;
      }
      return false;
    },
  });
}

export function useDownloadVideo() {
  return useMutation({
    mutationFn: ({
      videoId,
      onProgress,
    }: {
      videoId: string;
      onProgress?: (progress: number) => void;
    }) => videoService.downloadVideo(videoId, onProgress),
  });
}

export function useDeleteVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (videoId: string) => videoService.deleteVideo(videoId),
    onSuccess: (_, videoId) => {
      queryClient.removeQueries({ queryKey: ['video', videoId] });
      queryClient.invalidateQueries({ queryKey: ['videos'] });
    },
  });
}

export function useUploadScript() {
  return useMutation({
    mutationFn: ({
      file,
      onProgress,
    }: {
      file: { uri: string; type: string; name: string };
      onProgress?: (progress: number) => void;
    }) => videoService.uploadScript(file, onProgress),
  });
}


