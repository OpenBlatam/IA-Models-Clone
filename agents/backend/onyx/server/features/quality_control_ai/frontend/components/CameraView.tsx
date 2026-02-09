'use client';

import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { Play, Square, Camera as CameraIcon } from 'lucide-react';
import { qualityControlApi } from '@/lib/api';
import { useQualityControlStore } from '@/lib/store';
import { cn } from '@/lib/utils';
import { useToast } from '@/lib/hooks/useToast';
import Button from './ui/Button';
import Badge from './ui/Badge';
import LoadingSpinner from './LoadingSpinner';

const CameraView = memo((): JSX.Element => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { isInspecting, setInspecting, cameraInfo } = useQualityControlStore();
  const toast = useToast();

  const handleStart = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    try {
      await qualityControlApi.initializeCamera();
      const success = await qualityControlApi.startInspection();
      if (success) {
        setIsStreaming(true);
        setInspecting(true);
        toast.success('Inspection started successfully');
      } else {
        toast.error('Failed to start inspection');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start inspection';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  }, [setInspecting, toast]);

  const handleStop = useCallback(async (): Promise<void> => {
    try {
      await qualityControlApi.stopInspection();
      setIsStreaming(false);
      setInspecting(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      toast.info('Inspection stopped');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to stop inspection';
      toast.error(message);
    }
  }, [setInspecting, toast]);

  const handleCapture = useCallback(async (): Promise<void> => {
    setIsCapturing(true);
    try {
      const frame = await qualityControlApi.captureFrame();
      setCurrentFrame(frame);
      toast.success('Frame captured');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to capture frame';
      toast.error(message);
    } finally {
      setIsCapturing(false);
    }
  }, [toast]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent, action: () => void): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      action();
    }
  }, []);

  useEffect(() => {
    if (isStreaming) {
      intervalRef.current = setInterval(async () => {
        try {
          const frame = await qualityControlApi.captureFrame();
          setCurrentFrame(frame);
        } catch (error) {
          console.error('Failed to capture frame:', error);
        }
      }, 100); // Update every 100ms for ~10fps
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isStreaming]);

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Camera View</h2>
        <div className="flex items-center space-x-2">
          {cameraInfo ? (
            <Badge variant={cameraInfo.streaming ? 'success' : 'default'}>
              {cameraInfo.streaming ? 'Streaming' : 'Idle'}
            </Badge>
          ) : (
            <LoadingSpinner size="sm" />
          )}
        </div>
      </div>

      <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video mb-4">
        {currentFrame ? (
          <img
            src={currentFrame}
            alt="Camera feed"
            className="w-full h-full object-contain"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center">
              <CameraIcon className="w-16 h-16 text-gray-600 mx-auto mb-2" aria-hidden="true" />
              <p className="text-gray-400">No camera feed</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          {!isStreaming ? (
            <Button
              onClick={handleStart}
              onKeyDown={(e) => handleKeyDown(e, handleStart)}
              variant="primary"
              isLoading={isLoading}
              disabled={isLoading}
              tabIndex={0}
              aria-label="Start inspection"
            >
              <Play className="w-4 h-4" aria-hidden="true" />
              <span>Start Inspection</span>
            </Button>
          ) : (
            <Button
              onClick={handleStop}
              onKeyDown={(e) => handleKeyDown(e, handleStop)}
              variant="danger"
              tabIndex={0}
              aria-label="Stop inspection"
            >
              <Square className="w-4 h-4" aria-hidden="true" />
              <span>Stop Inspection</span>
            </Button>
          )}
          <Button
            onClick={handleCapture}
            onKeyDown={(e) => handleKeyDown(e, handleCapture)}
            variant="secondary"
            isLoading={isCapturing}
            disabled={isCapturing || !isStreaming}
            tabIndex={0}
            aria-label="Capture frame"
          >
            <CameraIcon className="w-4 h-4" aria-hidden="true" />
            <span>Capture</span>
          </Button>
        </div>
        {cameraInfo && (
          <div className="text-sm text-gray-600" aria-label="Camera resolution and frame rate">
            {cameraInfo.resolution.width}x{cameraInfo.resolution.height} @{' '}
            {Math.round(cameraInfo.fps)}fps
          </div>
        )}
      </div>
    </div>
  );
});

CameraView.displayName = 'CameraView';

export default CameraView;

