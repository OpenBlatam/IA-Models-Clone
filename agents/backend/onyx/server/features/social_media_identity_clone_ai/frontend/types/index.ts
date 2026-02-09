export enum Platform {
  TIKTOK = 'tiktok',
  INSTAGRAM = 'instagram',
  YOUTUBE = 'youtube',
}

export enum ContentType {
  VIDEO = 'video',
  POST = 'post',
  STORY = 'story',
  REEL = 'reel',
  COMMENT = 'comment',
}

export enum TaskStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical',
}

export enum PermissionLevel {
  VIEWER = 'viewer',
  EDITOR = 'editor',
  ADMIN = 'admin',
}

export interface VideoContent {
  video_id: string;
  url: string;
  title?: string;
  description?: string;
  transcript?: string;
  duration?: number;
  views?: number;
  likes?: number;
  comments?: number;
  created_at?: string;
  hashtags: string[];
  metadata: Record<string, unknown>;
}

export interface PostContent {
  post_id: string;
  url: string;
  caption?: string;
  image_urls: string[];
  likes?: number;
  comments?: number;
  created_at?: string;
  hashtags: string[];
  mentions: string[];
  metadata: Record<string, unknown>;
}

export interface CommentContent {
  comment_id: string;
  text: string;
  author?: string;
  likes?: number;
  created_at?: string;
  replies: CommentContent[];
}

export interface SocialProfile {
  platform: Platform;
  username: string;
  display_name?: string;
  bio?: string;
  profile_image_url?: string;
  followers_count?: number;
  following_count?: number;
  posts_count?: number;
  videos: VideoContent[];
  posts: PostContent[];
  comments: CommentContent[];
  extracted_at: string;
  metadata: Record<string, unknown>;
}

export interface ContentAnalysis {
  topics: string[];
  themes: string[];
  tone?: string;
  personality_traits: string[];
  communication_style?: string;
  common_phrases: string[];
  values: string[];
  interests: string[];
  language_patterns: Record<string, unknown>;
  sentiment_analysis: Record<string, number>;
}

export interface IdentityProfile {
  profile_id: string;
  username: string;
  display_name?: string;
  bio?: string;
  tiktok_profile?: SocialProfile;
  instagram_profile?: SocialProfile;
  youtube_profile?: SocialProfile;
  content_analysis: ContentAnalysis;
  knowledge_base: Record<string, unknown>;
  total_videos: number;
  total_posts: number;
  total_comments: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface GeneratedContent {
  content_id: string;
  identity_profile_id: string;
  platform: Platform;
  content_type: ContentType;
  content: string;
  title?: string;
  hashtags: string[];
  metadata: Record<string, unknown>;
  generated_at: string;
  confidence_score?: number;
}

export interface ExtractProfileRequest {
  platform: Platform;
  username: string;
  use_cache?: boolean;
}

export interface ExtractProfileResponse {
  success: boolean;
  platform: Platform;
  username: string;
  profile: SocialProfile;
  stats: {
    videos: number;
    posts: number;
    comments: number;
  };
}

export interface BuildIdentityRequest {
  tiktok_username?: string;
  instagram_username?: string;
  youtube_channel_id?: string;
}

export interface BuildIdentityResponse {
  success: boolean;
  identity_id: string;
  identity: IdentityProfile;
  stats: {
    total_videos: number;
    total_posts: number;
    total_comments: number;
    topics_count: number;
    themes_count: number;
  };
}

export interface GenerateContentRequest {
  identity_profile_id: string;
  platform: Platform;
  content_type: ContentType;
  topic?: string;
  style?: string;
  duration?: number;
  video_title?: string;
  tags?: string[];
}

export interface ContentValidation {
  is_valid: boolean;
  score: number;
  issues: string[];
  warnings: string[];
  suggestions: string[];
}

export interface GenerateContentResponse {
  success: boolean;
  content_id: string;
  content: GeneratedContent;
  validation: ContentValidation;
}

export interface Task {
  task_id: string;
  status: TaskStatus;
  result?: unknown;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface TaskResponse {
  success: boolean;
  task_id: string;
  status: TaskStatus;
}

export interface Metrics {
  counters: Record<string, number>;
  timers: Record<string, number>;
}

export interface MetricsResponse {
  success: boolean;
  metrics: Metrics;
}

export interface DashboardData {
  overview: {
    total_identities: number;
    total_content: number;
    content_today: number;
  };
  content_by_platform: Record<Platform, number>;
  recent_activity: unknown[];
  top_identities: IdentityProfile[];
}

export interface DashboardResponse {
  success: boolean;
  dashboard: DashboardData;
}

export interface Alert {
  alert_id: string;
  type: string;
  severity: AlertSeverity;
  message: string;
  acknowledged: boolean;
  created_at: string;
}

export interface AlertsResponse {
  success: boolean;
  count: number;
  critical_count: number;
  alerts: Alert[];
}

export interface Template {
  template_id: string;
  name: string;
  platform: Platform;
  content_type: ContentType;
  template: string;
  variables: string[];
  created_at: string;
}

export interface ABTest {
  test_id: string;
  name: string;
  variants: unknown[];
  status: string;
  results?: unknown;
}

export interface Recommendation {
  recommendation_id: string;
  type: string;
  title: string;
  description: string;
  priority: number;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}



