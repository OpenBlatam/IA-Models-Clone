'use client';

import { memo } from 'react';
import { useCamera } from '../hooks/useCamera';
import { useInspection } from '@/modules/inspection/hooks/useInspection';
import { useCameraStream } from '../hooks/useCameraStream';
import { useAsyncOperation } from '@/lib/hooks/useAsyncOperation';
import { useThrottle } from '@/lib/hooks/useThrottle';
import Card from '@/components/ui/Card';
import CameraFrame from './CameraFrame';
import CameraControls from './CameraControls';
import CameraInfo from './CameraInfo';

const CameraView = memo((): JSX.Element => {
  const { cameraInfo, initialize, captureFrame } = useCamera();
  const { start, stop } = useInspection();

  const { currentFrame, isStreaming, startStreaming, stopStreaming } = useCameraStream({
    captureFrame,
    interval: 100,
  });

  const throttledFrame = useThrottle(currentFrame, 100);

  const { execute: handleStart, isLoading } = useAsyncOperation(
    async () => {
      await initialize();
      const success = await start();
      if (success) {
        startStreaming();
        return success;
      }
      throw new Error('Failed to start inspection');
    },
    {
      successMessage: 'Inspection started successfully',
      errorMessage: 'Failed to start inspection',
    }
  );

  const { execute: handleStop } = useAsyncOperation(
    async () => {
      await stop();
      stopStreaming();
    },
    {
      successMessage: 'Inspection stopped',
      errorMessage: 'Failed to stop inspection',
    }
  );

  const { execute: handleCapture, isLoading: isCapturing } = useAsyncOperation(
    async () => {
      const frame = await captureFrame();
      return frame;
    },
    {
      successMessage: 'Frame captured',
      errorMessage: 'Failed to capture frame',
    }
  );

  return (
    <Card
      title="Camera View"
      headerActions={<CameraInfo cameraInfo={cameraInfo} />}
    >
      <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video mb-4">
        <CameraFrame frame={throttledFrame} isStreaming={isStreaming} />
      </div>

      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <CameraControls
          isStreaming={isStreaming}
          isLoading={isLoading}
          isCapturing={isCapturing}
          onStart={handleStart}
          onStop={handleStop}
          onCapture={handleCapture}
        />
        {cameraInfo && (
          <div className="text-sm text-gray-600" aria-label="Camera resolution and frame rate">
            {cameraInfo.resolution.width}x{cameraInfo.resolution.height} @{' '}
            {Math.round(cameraInfo.fps)}fps
          </div>
        )}
      </div>
    </Card>
  );
});

CameraView.displayName = 'CameraView';

export default CameraView;
