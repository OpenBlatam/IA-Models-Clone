/**
 * Video module specific types
 */

import type { VideoGenerationRequest, VideoGenerationResponse, VideoStatus } from '@/types/api';

export interface VideoCardProps {
  video: VideoGenerationResponse;
  onPress?: () => void;
  onLongPress?: () => void;
}

export interface VideoListProps {
  videos: VideoGenerationResponse[];
  isLoading?: boolean;
  onVideoPress?: (video: VideoGenerationResponse) => void;
  onVideoLongPress?: (video: VideoGenerationResponse) => void;
  emptyMessage?: string;
}

export interface VideoProgressProps {
  progress: {
    status: VideoStatus;
    progress: number;
    current_step: string;
    total_steps: number;
    completed_steps: number;
    estimated_time_remaining?: number;
    message?: string;
  };
  showDetails?: boolean;
}

export interface VideoStatusBadgeProps {
  status: VideoStatus;
  size?: 'small' | 'medium' | 'large';
}

export interface VideoGenerationFormData {
  script: string;
  language: string;
  videoStyle: string;
  audioVoice: string;
  subtitleStyle: string;
  subtitleEnabled: boolean;
}

export interface VideoFilters {
  status?: VideoStatus;
  dateFrom?: string;
  dateTo?: string;
  style?: string;
}

export interface VideoSortOptions {
  field: 'created_at' | 'duration' | 'status';
  direction: 'asc' | 'desc';
}

