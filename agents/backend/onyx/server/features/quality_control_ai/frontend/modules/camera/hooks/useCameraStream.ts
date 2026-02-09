import { useState, useRef, useCallback } from 'react';
import { useInterval } from '@/lib/hooks/useInterval';

interface UseCameraStreamOptions {
  captureFrame: () => Promise<string>;
  interval?: number;
}

export const useCameraStream = ({ captureFrame, interval = 100 }: UseCameraStreamOptions) => {
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const frameUpdateRef = useRef<number>(0);

  const updateFrame = useCallback(async (): Promise<void> => {
    frameUpdateRef.current += 1;
    const frameNumber = frameUpdateRef.current;

    try {
      const frame = await captureFrame();
      if (frameNumber === frameUpdateRef.current) {
        setCurrentFrame(frame);
      }
    } catch (error) {
      if (frameNumber === frameUpdateRef.current) {
        console.error('Failed to capture frame:', error);
      }
    }
  }, [captureFrame]);

  useInterval(updateFrame, {
    enabled: isStreaming,
    delay: isStreaming ? interval : null,
  });

  const startStreaming = useCallback(() => {
    setIsStreaming(true);
  }, []);

  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    frameUpdateRef.current = 0;
  }, []);

  return {
    currentFrame,
    isStreaming,
    startStreaming,
    stopStreaming,
  };
};

