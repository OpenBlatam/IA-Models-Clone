'use client';

import { useState, useCallback } from 'react';
import { Settings, Camera, Sliders } from 'lucide-react';
import { qualityControlApi } from '@/lib/api';
import { useQualityControlStore } from '@/lib/store';
import { useToast } from '@/lib/hooks/useToast';
import type { CameraSettings, DetectionSettings } from '@/lib/types';
import CameraSettingsModal from './CameraSettingsModal';
import DetectionSettingsModal from './DetectionSettingsModal';
import ReportGenerator from './ReportGenerator';
import Button from './ui/Button';

const ControlPanel = (): JSX.Element => {
  const [showCameraSettings, setShowCameraSettings] = useState(false);
  const [showDetectionSettings, setShowDetectionSettings] = useState(false);
  const [isInspectingFrame, setIsInspectingFrame] = useState(false);
  const { isInspecting, currentResult, setCurrentResult, addToHistory } = useQualityControlStore();
  const toast = useToast();

  const handleInspectFrame = useCallback(async (): Promise<void> => {
    if (!isInspecting) {
      toast.warning('Please start inspection first');
      return;
    }

    setIsInspectingFrame(true);
    try {
      const result = await qualityControlApi.inspectFrame();
      setCurrentResult(result);
      addToHistory(result);
      toast.success('Frame inspected successfully');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to inspect frame';
      toast.error(message);
    } finally {
      setIsInspectingFrame(false);
    }
  }, [isInspecting, setCurrentResult, addToHistory, toast]);

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
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Control Panel
      </h2>

      <div className="space-y-3">
        <Button
          onClick={handleInspectFrame}
          onKeyDown={(e) => handleKeyDown(e, handleInspectFrame)}
          variant="primary"
          isLoading={isInspectingFrame}
          disabled={!isInspecting || isInspectingFrame}
          className="w-full"
          tabIndex={0}
          aria-label="Inspect current frame"
        >
          <Camera className="w-4 h-4" aria-hidden="true" />
          <span>Inspect Frame</span>
        </Button>

        <Button
          onClick={() => setShowCameraSettings(true)}
          onKeyDown={(e) => handleKeyDown(e, () => setShowCameraSettings(true))}
          variant="secondary"
          className="w-full"
          tabIndex={0}
          aria-label="Camera settings"
        >
          <Settings className="w-4 h-4" aria-hidden="true" />
          <span>Camera Settings</span>
        </Button>

        <Button
          onClick={() => setShowDetectionSettings(true)}
          onKeyDown={(e) => handleKeyDown(e, () => setShowDetectionSettings(true))}
          variant="secondary"
          className="w-full"
          tabIndex={0}
          aria-label="Detection settings"
        >
          <Sliders className="w-4 h-4" aria-hidden="true" />
          <span>Detection Settings</span>
        </Button>
      </div>

      <div className="mt-6">
        <ReportGenerator />
      </div>

      {showCameraSettings && (
        <CameraSettingsModal
          onClose={() => setShowCameraSettings(false)}
          onSave={async (settings: CameraSettings) => {
            try {
              await qualityControlApi.updateCameraSettings(settings);
              toast.success('Camera settings updated');
              setShowCameraSettings(false);
            } catch (error) {
              const message = error instanceof Error ? error.message : 'Failed to update camera settings';
              toast.error(message);
            }
          }}
        />
      )}

      {showDetectionSettings && (
        <DetectionSettingsModal
          onClose={() => setShowDetectionSettings(false)}
          onSave={async (settings: DetectionSettings) => {
            try {
              await qualityControlApi.updateDetectionSettings(settings);
              toast.success('Detection settings updated');
              setShowDetectionSettings(false);
            } catch (error) {
              const message = error instanceof Error ? error.message : 'Failed to update detection settings';
              toast.error(message);
            }
          }}
        />
      )}
    </div>
  );
};

export default ControlPanel;

