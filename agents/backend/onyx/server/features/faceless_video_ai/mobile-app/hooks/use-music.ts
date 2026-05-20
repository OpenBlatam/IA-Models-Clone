import { useQuery } from '@tanstack/react-query';
import { musicService } from '@/services/music-service';

export function useMusicTracks(style?: string) {
  return useQuery({
    queryKey: ['music', 'tracks', style],
    queryFn: () => musicService.listTracks(style),
    staleTime: 300000, // 5 minutes
  });
}


