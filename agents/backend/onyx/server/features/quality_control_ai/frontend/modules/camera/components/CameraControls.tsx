'use client';

import { memo, useCallback } from 'react';
import { Play, Square, Camera as CameraIcon } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface CameraControlsProps {
  isStreaming: boolean;
  isLoading: boolean;
  isCapturing: boolean;
  onStart: () => void;
  onStop: () => void;
  onCapture: () => void;
}

const CameraControls = memo(
  ({
    isStreaming,
    isLoading,
    isCapturing,
    onStart,
    onStop,
    onCapture,
  }: CameraControlsProps): JSX.Element => {
    const handleKeyDown = useCallback(
      (e: React.KeyboardEvent, action: () => void): void => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          action();
        }
      },
      []
    );

    return (
      <div className="flex items-center space-x-2">
        {!isStreaming ? (
          <Button
            onClick={onStart}
            onKeyDown={(e) => handleKeyDown(e, onStart)}
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
            onClick={onStop}
            onKeyDown={(e) => handleKeyDown(e, onStop)}
            variant="danger"
            tabIndex={0}
            aria-label="Stop inspection"
          >
            <Square className="w-4 h-4" aria-hidden="true" />
            <span>Stop Inspection</span>
          </Button>
        )}
        <Button
          onClick={onCapture}
          onKeyDown={(e) => handleKeyDown(e, onCapture)}
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
    );
  }
);

CameraControls.displayName = 'CameraControls';

export default CameraControls;

