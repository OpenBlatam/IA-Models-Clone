/**
 * Hook for capturing screenshots from 3D view
 * @module robot-3d-view/hooks/use-screenshot
 */

import { useCallback, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import { captureScreenshot, downloadImage, copyImageToClipboard } from '../utils/screenshot-utils';

/**
 * Options for screenshot hook
 */
interface ScreenshotOptions {
  /** Image format */
  format?: 'image/png' | 'image/jpeg' | 'image/webp';
  /** Image quality */
  quality?: number;
  /** Auto-download */
  autoDownload?: boolean;
}

/**
 * Hook for capturing screenshots from 3D canvas
 * 
 * @param options - Screenshot options
 * @returns Screenshot capture functions
 * 
 * @example
 * ```tsx
 * const { capture, download, copy } = useScreenshot();
 * 
 * const handleCapture = async () => {
 *   const dataURL = await capture();
 *   download(dataURL, 'robot-view.png');
 * };
 * ```
 */
export function useScreenshot(options: ScreenshotOptions = {}) {
  const { gl } = useThree();
  const { format = 'image/png', quality = 0.92, autoDownload = false } = options;

  const capture = useCallback(async (): Promise<string> => {
    const canvas = gl.domElement;
    return captureScreenshot(canvas, { format, quality });
  }, [gl, format, quality]);

  const download = useCallback(
    async (filename?: string) => {
      const dataURL = await capture();
      downloadImage(dataURL, filename || `robot-3d-view-${Date.now()}.png`);
    },
    [capture]
  );

  const copy = useCallback(async () => {
    const dataURL = await capture();
    await copyImageToClipboard(dataURL);
  }, [capture]);

  return {
    capture,
    download,
    copy,
  };
}



