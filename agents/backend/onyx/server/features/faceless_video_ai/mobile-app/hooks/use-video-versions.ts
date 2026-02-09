import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { videoService } from '@/services/video-service';

export function useVideoVersions(videoId: string, enabled = true) {
  return useQuery({
    queryKey: ['video', videoId, 'versions'],
    queryFn: () => videoService.getVideoVersions(videoId),
    enabled: enabled && !!videoId,
    staleTime: 60000, // 1 minute
  });
}

export function useVideoVersion(videoId: string, versionNumber: number, enabled = true) {
  return useQuery({
    queryKey: ['video', videoId, 'version', versionNumber],
    queryFn: () => videoService.getVideoVersion(videoId, versionNumber),
    enabled: enabled && !!videoId && !!versionNumber,
    staleTime: 60000, // 1 minute
  });
}

export function useCompareVersions() {
  return useMutation({
    mutationFn: ({
      videoId,
      version1,
      version2,
    }: {
      videoId: string;
      version1: number;
      version2: number;
    }) => videoService.compareVersions(videoId, version1, version2),
  });
}


