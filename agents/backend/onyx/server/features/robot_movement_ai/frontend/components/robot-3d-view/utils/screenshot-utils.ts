/**
 * Screenshot and export utilities
 * @module robot-3d-view/utils/screenshot-utils
 */

/**
 * Options for screenshot capture
 */
interface ScreenshotOptions {
  /** Image format */
  format?: 'image/png' | 'image/jpeg' | 'image/webp';
  /** Image quality (0-1, for JPEG/WebP) */
  quality?: number;
  /** Filename */
  filename?: string;
}

/**
 * Captures a screenshot from a canvas element
 * 
 * @param canvas - Canvas element to capture
 * @param options - Screenshot options
 * @returns Promise that resolves to the image data URL
 */
export async function captureScreenshot(
  canvas: HTMLCanvasElement,
  options: ScreenshotOptions = {}
): Promise<string> {
  const { format = 'image/png', quality = 0.92 } = options;

  return new Promise((resolve, reject) => {
    try {
      const dataURL = canvas.toDataURL(format, quality);
      resolve(dataURL);
    } catch (error) {
      reject(error);
    }
  });
}

/**
 * Downloads an image from a data URL
 * 
 * @param dataURL - Image data URL
 * @param filename - Filename for download
 */
export function downloadImage(dataURL: string, filename: string = 'screenshot.png'): void {
  const link = document.createElement('a');
  link.href = dataURL;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Copies image to clipboard
 * 
 * @param dataURL - Image data URL
 * @returns Promise that resolves when image is copied
 */
export async function copyImageToClipboard(dataURL: string): Promise<void> {
  try {
    const response = await fetch(dataURL);
    const blob = await response.blob();
    await navigator.clipboard.write([
      new ClipboardItem({
        [blob.type]: blob,
      }),
    ]);
  } catch (error) {
    console.error('Failed to copy image to clipboard:', error);
    throw error;
  }
}

/**
 * Exports scene data as JSON
 * 
 * @param data - Scene data to export
 * @param filename - Filename for export
 */
export function exportSceneData(data: unknown, filename: string = 'scene-data.json'): void {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

