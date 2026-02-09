/**
 * Sharing utilities for configurations and recordings
 * @module robot-3d-view/utils/sharing
 */

import type { SceneConfig } from '../schemas/validation-schemas';
import type { Recording } from './recording';

/**
 * Share options
 */
export interface ShareOptions {
  title?: string;
  text?: string;
  url?: string;
}

/**
 * Shares configuration via Web Share API or clipboard
 * 
 * @param config - Configuration to share
 * @param options - Share options
 */
export async function shareConfig(
  config: SceneConfig,
  options: ShareOptions = {}
): Promise<void> {
  const { title = 'Robot 3D View Configuration', text, url } = options;

  const configJson = JSON.stringify(config, null, 2);
  const shareText = text || `${title}\n\n${configJson}`;

  if (navigator.share) {
    try {
      await navigator.share({
        title,
        text: shareText,
        url,
      });
      return;
    } catch (error) {
      // User cancelled or error occurred, fall back to clipboard
    }
  }

  // Fallback to clipboard
  await navigator.clipboard.writeText(shareText);
}

/**
 * Shares recording via Web Share API or download
 * 
 * @param recording - Recording to share
 * @param options - Share options
 */
export async function shareRecording(
  recording: Recording,
  options: ShareOptions = {}
): Promise<void> {
  const { title = recording.name, text, url } = options;

  const recordingJson = JSON.stringify(recording, null, 2);
  const shareText = text || `${title}\n\n${recordingJson}`;

  if (navigator.share) {
    try {
      await navigator.share({
        title,
        text: shareText,
        url,
      });
      return;
    } catch (error) {
      // User cancelled or error occurred, fall back to download
    }
  }

  // Fallback to download
  const blob = new Blob([recordingJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${recording.name}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Generates a shareable URL with encoded configuration
 * 
 * @param config - Configuration to encode
 * @param baseUrl - Base URL
 * @returns Shareable URL
 */
export function generateShareableUrl(
  config: SceneConfig,
  baseUrl = window.location.origin
): string {
  const encoded = btoa(JSON.stringify(config));
  return `${baseUrl}?config=${encoded}`;
}

/**
 * Parses configuration from shareable URL
 * 
 * @param url - URL to parse
 * @returns Configuration or null
 */
export function parseShareableUrl(url: string): SceneConfig | null {
  try {
    const urlObj = new URL(url);
    const encoded = urlObj.searchParams.get('config');
    if (!encoded) return null;

    const decoded = atob(encoded);
    return JSON.parse(decoded) as SceneConfig;
  } catch (error) {
    return null;
  }
}

/**
 * Copies text to clipboard
 * 
 * @param text - Text to copy
 */
export async function copyToClipboard(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
}

/**
 * Generates a QR code data URL for sharing
 * 
 * @param data - Data to encode
 * @returns QR code data URL (requires QR code library)
 */
export async function generateQRCode(data: string): Promise<string> {
  // This would require a QR code library like 'qrcode'
  // For now, return a placeholder
  throw new Error('QR code generation requires a QR code library');
}



