/**
 * Screenshot Controls Component
 * @module robot-3d-view/controls/screenshot-controls
 */

'use client';

import { memo } from 'react';
import { useScreenshot } from '../hooks/use-screenshot';

/**
 * Screenshot Controls Component
 * 
 * Provides UI controls for capturing and exporting screenshots.
 * 
 * @returns Screenshot controls component
 */
export const ScreenshotControls = memo(() => {
  const { capture, download, copy } = useScreenshot();

  const handleCapture = async () => {
    try {
      await download();
    } catch (error) {
      console.error('Failed to capture screenshot:', error);
    }
  };

  const handleCopy = async () => {
    try {
      await copy();
      // Could show a toast notification here
    } catch (error) {
      console.error('Failed to copy screenshot:', error);
    }
  };

  return (
    <div className="absolute bottom-4 right-4 flex flex-col gap-2">
      <button
        onClick={handleCapture}
        className="px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg"
        title="Capturar screenshot"
      >
        📸 Capturar
      </button>
      <button
        onClick={handleCopy}
        className="px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg"
        title="Copiar al portapapeles"
      >
        📋 Copiar
      </button>
    </div>
  );
});

ScreenshotControls.displayName = 'ScreenshotControls';



