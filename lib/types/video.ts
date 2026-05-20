export interface VideoPlayerProps {
  videoUrl: string;
  autoPlay?: boolean;
  volume?: number;
  onTimeUpdate?: (currentTime: number) => void;
  onEnded?: () => void;
  onProgress?: (progress: number) => void;
  onError?: (error: string) => void;
  onVolumeChange?: (volume: number) => void;
  className?: string;
}

export interface VideoControlsProps {
  isPlaying: boolean;
  isMuted: boolean;
  volume: number;
  currentTime: number;
  duration: number;
  isFullscreen: boolean;
  showControls: boolean;
  playbackRate: number;
  onPlayPause: () => void;
  onMute: () => void;
  onVolumeChange: (volume: number) => void;
  onSeek: (time: number) => void;
  onPlaybackRateChange: (rate: number) => void;
  onFullscreen: () => void;
  className?: string;
}

export interface VideoState {
  isPlaying: boolean;
  isMuted: boolean;
  currentVolume: number;
  currentTime: number;
  duration: number;
  isFullscreen: boolean;
  showControls: boolean;
  isLoading: boolean;
  error: string | null;
  playbackRate: number;
  isDragging: boolean;
  previewTime: number | null;
}

export interface VideoComment {
  id: string;
  content: string;
  user: {
    id: string;
    name: string;
    image?: string;
  };
  createdAt: string;
  updatedAt: string;
  likes: number;
  isLiked: boolean;
  replies?: VideoComment[];
}

export interface VideoQuestion {
  id: string;
  content: string;
  user: {
    id: string;
    name: string;
    image?: string;
  };
  createdAt: string;
  updatedAt: string;
  status: 'pending' | 'answered' | 'error';
  answer?: string;
}

export interface VideoResource {
  id: string;
  title: string;
  type: 'file' | 'reading';
  url: string;
}

export interface VideoData {
  comments: VideoComment[];
  resources: VideoResource[];
  questions?: VideoQuestion[];
}

export interface VideoUploadProgress {
  progress: number;
  isUploading: boolean;
  error?: string;
}

export interface VideoMetadata {
  id: string;
  title: string;
  description?: string;
  duration: number;
  thumbnail?: string;
  url: string;
  createdAt: string;
  updatedAt: string;
}

export interface PlaylistItem {
  id: string;
  title: string;
  duration: number;
  thumbnail?: string;
  url: string;
  isCompleted?: boolean;
  progress?: number;
}

export interface VideoAnalytics {
  views: number;
  watchTime: number;
  completionRate: number;
  engagementRate: number;
  averageWatchTime: number;
}
