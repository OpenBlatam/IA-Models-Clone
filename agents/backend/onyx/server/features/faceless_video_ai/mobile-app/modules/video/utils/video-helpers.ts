/**
 * Video helper utilities
 */

import type { VideoGenerationResponse, VideoStatus } from '@/types/api';
import { VIDEO_STATUS_COLORS } from '@/utils/constants';

export function getVideoStatusColor(status: VideoStatus): string {
  return VIDEO_STATUS_COLORS[status] || '#666666';
}

export function isVideoProcessing(status: VideoStatus): boolean {
  return [
    'pending',
    'processing',
    'generating_images',
    'generating_audio',
    'adding_subtitles',
    'compositing',
  ].includes(status);
}

export function isVideoCompleted(status: VideoStatus): boolean {
  return status === 'completed';
}

export function isVideoFailed(status: VideoStatus): boolean {
  return status === 'failed';
}

export function canDownloadVideo(video: VideoGenerationResponse): boolean {
  return isVideoCompleted(video.status) && !!video.video_url;
}

export function canDeleteVideo(video: VideoGenerationResponse): boolean {
  return true; // Can always delete
}

export function canShareVideo(video: VideoGenerationResponse): boolean {
  return isVideoCompleted(video.status);
}

export function getVideoThumbnail(video: VideoGenerationResponse): string | null {
  return video.thumbnail_url || null;
}

export function formatVideoSize(bytes?: number): string {
  if (!bytes) return 'Unknown';
  
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function estimateVideoGenerationTime(scriptLength: number): number {
  // Rough estimate: ~1 second per 10 characters
  return Math.max(30, Math.ceil(scriptLength / 10));
}

