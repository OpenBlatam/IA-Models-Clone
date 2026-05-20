import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

export const usePrefetch = () => {
  const queryClient = useQueryClient();

  const prefetchCameraInfo = useCallback(() => {
    queryClient.prefetchQuery({
      queryKey: ['camera-info'],
      queryFn: async () => {
        const { cameraApi } = await import('@/modules/camera/api');
        return cameraApi.getInfo();
      },
    });
  }, [queryClient]);

  const prefetchAlerts = useCallback(() => {
    queryClient.prefetchQuery({
      queryKey: ['alerts'],
      queryFn: async () => {
        const { alertsApi } = await import('@/modules/alerts/api');
        return alertsApi.getRecent(undefined, 50);
      },
    });
  }, [queryClient]);

  return {
    prefetchCameraInfo,
    prefetchAlerts,
  };
};

