import { useCallback } from 'react';
import { inspectionApi } from '../api';
import { useQualityControlStore } from '@/lib/store';
import { useToast } from '@/lib/hooks/useToast';
import type { InspectionResult } from '../types';

export const useInspection = () => {
  const { setInspecting, setCurrentResult, addToHistory, isInspecting } = useQualityControlStore();
  const toast = useToast();

  const start = useCallback(async (): Promise<boolean> => {
    try {
      const success = await inspectionApi.start();
      if (success) {
        setInspecting(true);
        toast.success('Inspection started successfully');
      } else {
        toast.error('Failed to start inspection');
      }
      return success;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start inspection';
      toast.error(message);
      return false;
    }
  }, [setInspecting, toast]);

  const stop = useCallback(async (): Promise<void> => {
    try {
      await inspectionApi.stop();
      setInspecting(false);
      toast.info('Inspection stopped');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to stop inspection';
      toast.error(message);
    }
  }, [setInspecting, toast]);

  const inspectFrame = useCallback(
    async (image?: string): Promise<InspectionResult | null> => {
      try {
        const result = await inspectionApi.inspectFrame(image);
        setCurrentResult(result);
        addToHistory(result);
        return result;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to inspect frame';
        toast.error(message);
        return null;
      }
    },
    [setCurrentResult, addToHistory, toast]
  );

  const inspectBatch = useCallback(
    async (images: string[]): Promise<InspectionResult[]> => {
      try {
        return await inspectionApi.inspectBatch({ images });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to inspect batch';
        toast.error(message);
        return [];
      }
    },
    [toast]
  );

  return {
    isInspecting,
    start,
    stop,
    inspectFrame,
    inspectBatch,
  };
};

