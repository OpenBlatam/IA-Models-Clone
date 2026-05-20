import { useState, useCallback, useRef } from 'react';

interface UseVideoControlsOptions {
  autoHideDelay?: number;
}

export function useVideoControls({ autoHideDelay = 3000 }: UseVideoControlsOptions = {}) {
  const [showControls, setShowControls] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [previewTime, setPreviewTime] = useState<number | null>(null);
  const controlsTimeoutRef = useRef<NodeJS.Timeout>();

  const showControlsTemporarily = useCallback(() => {
    setShowControls(true);
    
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    
    controlsTimeoutRef.current = setTimeout(() => {
      setShowControls(false);
    }, autoHideDelay);
  }, [autoHideDelay]);

  const handleMouseMove = useCallback(() => {
    if (!isDragging) {
      showControlsTemporarily();
    }
  }, [isDragging, showControlsTemporarily]);

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setShowControls(false);
    }
  }, [isDragging]);

  const startDragging = useCallback(() => {
    setIsDragging(true);
    setShowControls(true);
  }, []);

  const stopDragging = useCallback(() => {
    setIsDragging(false);
    setPreviewTime(null);
    showControlsTemporarily();
  }, [showControlsTemporarily]);

  const updatePreviewTime = useCallback((time: number) => {
    setPreviewTime(time);
  }, []);

  const clearPreviewTime = useCallback(() => {
    setPreviewTime(null);
  }, []);

  return {
    showControls,
    isDragging,
    previewTime,
    actions: {
      showControlsTemporarily,
      handleMouseMove,
      handleMouseLeave,
      startDragging,
      stopDragging,
      updatePreviewTime,
      clearPreviewTime,
      setShowControls,
    },
  };
}
