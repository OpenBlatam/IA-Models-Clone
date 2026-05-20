// API Types matching backend models

export const VideoStatus = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  GENERATING_IMAGES: 'generating_images',
  GENERATING_AUDIO: 'generating_audio',
  ADDING_SUBTITLES: 'adding_subtitles',
  COMPOSITING: 'compositing',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const;

export type VideoStatus = (typeof VideoStatus)[keyof typeof VideoStatus];

export const SubtitleStyle = {
  SIMPLE: 'simple',
  MODERN: 'modern',
  BOLD: 'bold',
  ELEGANT: 'elegant',
  MINIMAL: 'minimal',
} as const;

export type SubtitleStyle = (typeof SubtitleStyle)[keyof typeof SubtitleStyle];

export const VideoStyle = {
  REALISTIC: 'realistic',
  ANIMATED: 'animated',
  ABSTRACT: 'abstract',
  MINIMALIST: 'minimalist',
  DYNAMIC: 'dynamic',
} as const;

export type VideoStyle = (typeof VideoStyle)[keyof typeof VideoStyle];

export const AudioVoice = {
  MALE_1: 'male_1',
  MALE_2: 'male_2',
  FEMALE_1: 'female_1',
  FEMALE_2: 'female_2',
  NEUTRAL: 'neutral',
} as const;

export type AudioVoice = (typeof AudioVoice)[keyof typeof AudioVoice];

export interface SubtitleConfig {
  enabled: boolean;
  style: SubtitleStyle;
  font_size?: number;
  font_color?: string;
  background_color?: string;
  position?: 'top' | 'center' | 'bottom';
  animation?: boolean;
  max_chars_per_line?: number;
  fade_in?: boolean;
  fade_out?: boolean;
}

export interface AudioConfig {
  voice: AudioVoice;
  speed?: number;
  pitch?: number;
  volume?: number;
  background_music?: boolean;
  music_volume?: number;
  music_style?: string;
}

export interface VideoConfig {
  resolution?: string;
  fps?: number;
  duration?: number;
  style: VideoStyle;
  transition_duration?: number;
  image_duration?: number;
  background_color?: string;
}

export interface VideoScript {
  text: string;
  segments?: Array<Record<string, unknown>>;
  language?: string;
  metadata?: Record<string, unknown>;
}

export interface GenerationProgress {
  status: VideoStatus;
  progress: number;
  current_step: string;
  total_steps: number;
  completed_steps: number;
  estimated_time_remaining?: number;
  message?: string;
}

export interface VideoGenerationRequest {
  script: VideoScript;
  video_config?: VideoConfig;
  audio_config?: AudioConfig;
  subtitle_config?: SubtitleConfig;
  output_format?: 'mp4' | 'webm' | 'mov';
  output_quality?: 'low' | 'medium' | 'high' | 'ultra';
}

export interface VideoGenerationResponse {
  video_id: string;
  status: VideoStatus;
  progress: GenerationProgress;
  video_url?: string;
  thumbnail_url?: string;
  duration?: number;
  file_size?: number;
  created_at: string;
  error?: string;
  metadata?: Record<string, unknown>;
}

// Auth Types
export interface RegisterRequest {
  email: string;
  password: string;
  roles?: string[];
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface User {
  user_id: string;
  email: string;
  roles: string[];
  api_key?: string;
  created_at?: string;
}

// Template Types
export interface Template {
  name: string;
  description: string;
  config: {
    video_config: VideoConfig;
    audio_config: AudioConfig;
    subtitle_config: SubtitleConfig;
  };
}

export interface CustomTemplate extends Template {
  template_id: string;
  user_id: string;
  is_public: boolean;
  created_at: string;
}

// Batch Types
export interface BatchGenerationRequest {
  requests: VideoGenerationRequest[];
  webhook_url?: string;
}

export interface BatchGenerationResponse {
  batch_id: string;
  video_ids: string[];
  status: 'processing' | 'completed' | 'failed';
}

// Analytics Types
export interface Analytics {
  metrics: {
    total_videos: number;
    completed_videos: number;
    failed_videos: number;
    average_generation_time: number;
  };
  usage_statistics: {
    videos_per_day: Array<{ date: string; count: number }>;
    videos_by_style: Record<string, number>;
    videos_by_voice: Record<string, number>;
  };
  top_errors: Array<{ error: string; count: number }>;
}

// Search Types
export interface SearchFilters {
  status?: VideoStatus;
  tags?: string[];
  date_from?: string;
  date_to?: string;
}

export interface SearchResult {
  video_id: string;
  status: VideoStatus;
  created_at: string;
  duration?: number;
  thumbnail_url?: string;
}

// Feedback Types
export interface Feedback {
  feedback_id: string;
  video_id: string;
  user_id: string;
  rating: number;
  comment?: string;
  tags?: string[];
  created_at: string;
}

export interface FeedbackStats {
  average_rating: number;
  total_feedbacks: number;
  rating_distribution: Record<number, number>;
}

// Platform Export Types
export interface PlatformSpec {
  name: string;
  resolution: string;
  fps: number;
  format: string;
  max_duration?: number;
  max_file_size?: number;
}

// Music Library Types
export interface MusicTrack {
  track_id: string;
  name: string;
  style: string;
  duration: number;
  url: string;
}

// Recommendations Types
export interface Recommendations {
  video_config: VideoConfig;
  audio_config: AudioConfig;
  subtitle_config: SubtitleConfig;
  reasoning: string;
  confidence: number;
}

// Share Types
export interface Share {
  share_id: string;
  video_id: string;
  owner_id: string;
  shared_with_id?: string;
  shared_with_email?: string;
  permission: 'view' | 'edit' | 'admin';
  is_public: boolean;
  share_token?: string;
  expires_at?: string;
  created_at: string;
}

// Schedule Types
export interface ScheduledJob {
  job_id: string;
  video_id: string;
  scheduled_at: string;
  timezone: string;
  repeat?: string;
  status: 'pending' | 'completed' | 'cancelled';
  created_at: string;
}

// Quota Types
export interface QuotaInfo {
  user_id: string;
  videos_generated: number;
  videos_limit: number;
  storage_used: number;
  storage_limit: number;
  reset_at: string;
}

// Admin Types
export interface AdminDashboard {
  total_videos: number;
  total_users: number;
  system_health: string;
  recent_activity: Array<{
    type: string;
    timestamp: string;
    details: Record<string, unknown>;
  }>;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: Array<{
    name: string;
    status: 'pass' | 'fail';
    message?: string;
  }>;
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_connections: number;
  requests_per_minute: number;
}

// Alert Types
export interface Alert {
  alert_id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  acknowledged: boolean;
  resolved: boolean;
  created_at: string;
}

// API Error Types
export interface ApiError {
  detail: string;
  status_code: number;
  error_code?: string;
}

// Rate Limit Types
export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset_at: number;
}


