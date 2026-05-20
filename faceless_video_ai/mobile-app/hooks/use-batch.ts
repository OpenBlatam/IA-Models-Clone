import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { batchService } from '@/services/video-service';
import type { BatchGenerationRequest } from '@/types/api';

export function useBatchGenerate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BatchGenerationRequest) =>
      batchService.generateBatch(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
      queryClient.invalidateQueries({ queryKey: ['batch'] });
    },
  });
}

export function useBatchStatus(videoIds: string[], enabled = true) {
  return useQuery({
    queryKey: ['batch', 'status', videoIds.sort().join(',')],
    queryFn: () => batchService.getBatchStatus(videoIds),
    enabled: enabled && videoIds.length > 0,
    refetchInterval: 5000, // Poll every 5 seconds
  });
}


