'use client';

import { useState, useCallback, memo } from 'react';
import { useToggle } from '@/lib/hooks/useToggle';
import { Settings, Camera, Sliders } from 'lucide-react';
import { useInspection } from '@/modules/inspection/hooks/useInspection';
import { useCamera } from '@/modules/camera/hooks/useCamera';
import { useQualityControlStore } from '@/lib/store';
import { useToast } from '@/lib/hooks/useToast';
import type { CameraSettings } from '@/modules/camera/types';
import type { DetectionSettings } from '@/modules/detection/types';
import CameraSettingsModal from '@/modules/camera/components/CameraSettingsModal';
import DetectionSettingsModal from '@/modules/detection/components/DetectionSettingsModal';
import ReportGenerator from '@/modules/reports/components/ReportGenerator';
import { detectionApi } from '@/modules/detection/api';
import { Button } from '@/components/ui/Button';
import Card from '@/components/ui/Card';

const ControlPanel = memo((): JSX.Element => {
  const {
    value: showCameraSettings,
    setTrue: openCameraSettings,
    setFalse: closeCameraSettings,
  } = useToggle();
  const {
    value: showDetectionSettings,
    setTrue: openDetectionSettings,
    setFalse: closeDetectionSettings,
  } = useToggle();
  const { isInspecting, setCurrentResult, addToHistory } = useQualityControlStore();
  const { inspectFrame } = useInspection();
  const { updateSettings: updateCameraSettings } = useCamera();
  const toast = useToast();
  const [isInspectingFrame, setIsInspectingFrame] = useState(false);

  const handleInspectFrame = useCallback(async (): Promise<void> => {
    if (!isInspecting) {
      toast.warning('Please start inspection first');
      return;
    }

    setIsInspectingFrame(true);
    try {
      const result = await inspectFrame();
      if (result) {
        setCurrentResult(result);
        addToHistory(result);
        toast.success('Frame inspected successfully');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to inspect frame';
      toast.error(message);
    } finally {
      setIsInspectingFrame(false);
    }
  }, [isInspecting, inspectFrame, setCurrentResult, addToHistory, toast]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, action: () => void): void => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        action();
      }
    },
    []
  );

  const handleSaveCameraSettings = useCallback(
    async (settings: CameraSettings): Promise<void> => {
      await updateCameraSettings(settings);
      closeCameraSettings();
    },
    [updateCameraSettings, closeCameraSettings]
  );

  const handleSaveDetectionSettings = useCallback(
    async (settings: DetectionSettings): Promise<void> => {
      await detectionApi.updateSettings(settings);
      closeDetectionSettings();
    },
    [closeDetectionSettings]
  );

  return (
    <Card title="Control Panel">
      <div className="space-y-3">
        <Button
          onClick={() => handleInspectFrame()}
          onKeyDown={(e) => handleKeyDown(e, () => handleInspectFrame())}
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
          onClick={openCameraSettings}
          onKeyDown={(e) => handleKeyDown(e, openCameraSettings)}
          variant="secondary"
          className="w-full"
          tabIndex={0}
          aria-label="Camera settings"
        >
          <Settings className="w-4 h-4" aria-hidden="true" />
          <span>Camera Settings</span>
        </Button>

        <Button
          onClick={openDetectionSettings}
          onKeyDown={(e) => handleKeyDown(e, openDetectionSettings)}
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

      <CameraSettingsModal
        open={showCameraSettings}
        onClose={closeCameraSettings}
        onSave={handleSaveCameraSettings}
      />

      <DetectionSettingsModal
        open={showDetectionSettings}
        onClose={closeDetectionSettings}
        onSave={handleSaveDetectionSettings}
      />
    </Card>
  );
});

ControlPanel.displayName = 'ControlPanel';

export default ControlPanel;
