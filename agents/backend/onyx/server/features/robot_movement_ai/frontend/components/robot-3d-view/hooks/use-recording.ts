/**
 * Hook for recording management
 * @module robot-3d-view/hooks/use-recording
 */

import { useState, useEffect, useCallback } from 'react';
import { recordingManager, type Recording } from '../utils/recording';
import type { Position3D, SceneConfig } from '../schemas/validation-schemas';

/**
 * Hook for managing recordings
 * 
 * @returns Recording state and actions
 * 
 * @example
 * ```tsx
 * const { isRecording, startRecording, stopRecording, recordFrame } = useRecording();
 * ```
 */
export function useRecording() {
  const [isRecording, setIsRecording] = useState(() =>
    recordingManager.getIsRecording()
  );
  const [recordings, setRecordings] = useState<readonly Recording[]>(() =>
    recordingManager.getRecordings()
  );

  useEffect(() => {
    const interval = setInterval(() => {
      setIsRecording(recordingManager.getIsRecording());
      setRecordings(recordingManager.getRecordings());
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const startRecording = useCallback((name?: string) => {
    recordingManager.startRecording(name);
    setIsRecording(true);
    setRecordings(recordingManager.getRecordings());
  }, []);

  const stopRecording = useCallback((): Recording | null => {
    const recording = recordingManager.stopRecording();
    setIsRecording(false);
    setRecordings(recordingManager.getRecordings());
    return recording;
  }, []);

  const recordFrame = useCallback(
    (
      currentPos: Position3D,
      targetPos: Position3D | null,
      config: SceneConfig
    ) => {
      recordingManager.recordFrame(currentPos, targetPos, config);
    },
    []
  );

  const deleteRecording = useCallback((id: string) => {
    recordingManager.deleteRecording(id);
    setRecordings(recordingManager.getRecordings());
  }, []);

  const exportRecording = useCallback((id: string, filename?: string) => {
    recordingManager.exportRecording(id, filename);
  }, []);

  const importRecording = useCallback(async (file: File) => {
    return recordingManager.importRecording(file);
  }, []);

  return {
    isRecording,
    recordings,
    startRecording,
    stopRecording,
    recordFrame,
    deleteRecording,
    exportRecording,
    importRecording,
  };
}



