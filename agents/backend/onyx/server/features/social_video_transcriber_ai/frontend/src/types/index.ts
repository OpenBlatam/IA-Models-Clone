// Types based on Python models

export enum SupportedPlatform {
  TIKTOK = 'tiktok',
  INSTAGRAM = 'instagram',
  YOUTUBE = 'youtube',
  AUTO = 'auto',
}

export enum TranscriptionStatus {
  PENDING = 'pending',
  DOWNLOADING = 'downloading',
  EXTRACTING_AUDIO = 'extracting_audio',
  TRANSCRIBING = 'transcribing',
  ANALYZING = 'analyzing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

export enum ContentFramework {
  HOOK_STORY_OFFER = 'hook_story_offer',
  PROBLEM_AGITATE_SOLVE = 'problem_agitate_solve',
  AIDA = 'aida',
  STAR = 'star',
  BAB = 'bab',
  EDUCATIONAL = 'educational',
  STORYTELLING = 'storytelling',
  LISTICLE = 'listicle',
  TUTORIAL = 'tutorial',
  REVIEW = 'review',
  NEWS = 'news',
  ENTERTAINMENT = 'entertainment',
  MOTIVATIONAL = 'motivational',
  CUSTOM = 'custom',
}

export interface TranscriptionSegment {
  id: number
  start_time: number
  end_time: number
  text: string
  confidence?: number
  speaker?: string
  formatted_timestamp?: string
  duration?: number
}

export interface ContentAnalysis {
  framework: ContentFramework
  framework_confidence: number
  structure: Record<string, any>
  key_points: string[]
  tone: string
  target_audience?: string
  call_to_action?: string
  hashtags_suggested: string[]
  content_type: string
  language_detected: string
  word_count: number
  estimated_reading_time: number
}

export interface TranscriptionRequest {
  video_url: string
  platform?: SupportedPlatform
  include_timestamps?: boolean
  include_analysis?: boolean
  language?: string
  webhook_url?: string
}

export interface TranscriptionResponse {
  job_id: string
  status: TranscriptionStatus
  platform_detected?: SupportedPlatform
  video_title?: string
  video_duration?: number
  video_author?: string
  full_text?: string
  full_text_with_timestamps?: string
  segments?: TranscriptionSegment[]
  analysis?: ContentAnalysis
  created_at: string
  completed_at?: string
  processing_time?: number
  error?: string
}

export interface VariantRequest {
  text?: string
  job_id?: string
  num_variants?: number
  preserve_structure?: boolean
  preserve_length?: boolean
  target_tone?: string
  custom_instructions?: string
}

export interface TextVariant {
  id: string
  original_text: string
  variant_text: string
  variant_type: string
  similarity_score: number
  preserves_structure: boolean
  preserves_length: boolean
  tone?: string
  word_count: number
  created_at: string
}

export interface VariantResponse {
  request_id: string
  original_text: string
  variants: TextVariant[]
  analysis?: ContentAnalysis
  created_at: string
  processing_time?: number
}

export interface AnalysisRequest {
  text: string
  analyze_framework?: boolean
  analyze_structure?: boolean
  suggest_improvements?: boolean
}

export interface AnalysisResponse {
  request_id: string
  analysis: ContentAnalysis
  improvements?: string[]
  created_at: string
}

export interface QuickVariantRequest {
  job_id: string
}

export interface BatchJob {
  batch_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  total_urls: number
  completed_urls: number
  failed_urls: number
  created_at: string
  completed_at?: string
  results?: Record<string, TranscriptionResponse>
  errors?: string[]
}

export interface BatchRequest {
  urls: string[]
  include_timestamps?: boolean
  include_analysis?: boolean
  language?: string
  webhook_url?: string
}

export interface VideoInfo {
  title: string
  duration: number
  author: string
  platform: SupportedPlatform
  thumbnail_url?: string
  description?: string
}

export interface Platform {
  name: string
  id: string
  supported_urls: string[]
}

export interface JobListItem {
  job_id: string
  status: TranscriptionStatus
  video_title?: string
  platform?: SupportedPlatform
  created_at: string
  completed_at?: string
}

export interface JobsListResponse {
  jobs: JobListItem[]
  total: number
}

export interface CacheStats {
  total_entries: number
  hit_rate: number
  miss_rate: number
  total_hits: number
  total_misses: number
}

export interface UserTier {
  name: string
  limits: {
    requests_per_minute: number
    requests_per_hour: number
    requests_per_day: number
    max_video_duration_seconds: number
    max_batch_size: number
  }
}

export interface UsageStats {
  requests_today: number
  requests_this_hour: number
  requests_this_minute: number
  total_requests: number
  tier: string
  limits: {
    requests_per_minute: number
    requests_per_hour: number
    requests_per_day: number
  }
}

export interface HealthResponse {
  status: string
  service: string
  version: string
  active_jobs: number
  features: string[]
}

export interface Keyword {
  keyword: string
  relevance_score: number
  category: string
}

export interface KeywordsResponse {
  keywords: Keyword[]
  count: number
}

export interface SummaryResponse {
  brief_summary: string
  detailed_summary: string
  bullet_points: string[]
  main_topic: string
  subtopics: string[]
}

export interface SentimentResponse {
  overall_sentiment: 'positive' | 'negative' | 'neutral'
  sentiment_score: number
  emotions_detected: string[]
  tone_descriptors: string[]
  confidence: number
}

export interface FullAnalysisResponse {
  keywords: Keyword[]
  summary: SummaryResponse
  sentiment: SentimentResponse
  framework_analysis: ContentAnalysis
}




