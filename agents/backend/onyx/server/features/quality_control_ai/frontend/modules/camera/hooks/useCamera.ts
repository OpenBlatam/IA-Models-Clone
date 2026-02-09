import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { cameraApi } from '../api';
import { useQualityControlStore } from '@/lib/store';
import { CAMERA_REFRESH_INTERVAL } from '@/config/constants';

export const useCamera = () => {
  const { setCameraInfo, cameraInfo: storeCameraInfo } = useQualityControlStore();

  const { data, isLoading, error } = useQuery({
    queryKey: ['camera-info'],
    queryFn: () => cameraApi.getInfo(),
    refetchInterval: CAMERA_REFRESH_INTERVAL,
    staleTime: 1000,
    select: (data) => data,
  });

  const cameraInfo = useMemo(() => data || storeCameraInfo, [data, storeCameraInfo]);

  const initialize = async (): Promise<boolean> => {
    try {
      return await cameraApi.initialize();
    } catch {
      return false;
    }
  };

  const updateSettings = async (
    settings: Parameters<typeof cameraApi.updateSettings>[0]
  ): Promise<void> => {
    await cameraApi.updateSettings(settings);
  };

  const captureFrame = async (): Promise<string> => {
    return await cameraApi.captureFrame();
  };

  return {
    cameraInfo,
    isLoading,
    error,
    initialize,
    updateSettings,
    captureFrame,
  };
};
