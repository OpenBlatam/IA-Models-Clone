/**
 * TypeScript types matching backend schemas
 */

export enum SocialMediaPlatform {
  FACEBOOK = 'facebook',
  TWITTER = 'twitter',
  INSTAGRAM = 'instagram',
  LINKEDIN = 'linkedin',
  TIKTOK = 'tiktok',
  YOUTUBE = 'youtube',
  REDDIT = 'reddit',
  DISCORD = 'discord',
  TELEGRAM = 'telegram',
}

export enum ConnectionStatus {
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  EXPIRED = 'expired',
  ERROR = 'error',
}

export enum ValidationStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface SocialMediaConnectRequest {
  platform: SocialMediaPlatform;
  access_token: string;
  refresh_token?: string;
  expires_in?: number;
}

export interface SocialMediaConnectionResponse {
  id: string;
  platform: SocialMediaPlatform;
  status: ConnectionStatus;
  connected_at?: string;
  profile_data: Record<string, unknown>;
}

export interface ValidationCreate {
  platforms: SocialMediaPlatform[];
  include_historical_data?: boolean;
  analysis_depth?: 'basic' | 'standard' | 'deep';
}

export interface ValidationRead {
  id: string;
  user_id: string;
  status: ValidationStatus;
  connected_platforms: SocialMediaPlatform[];
  created_at: string;
  updated_at: string;
  completed_at?: string;
  has_profile: boolean;
  has_report: boolean;
}

export interface PersonalityTraits {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface PsychologicalProfileResponse {
  id: string;
  user_id: string;
  personality_traits: PersonalityTraits;
  emotional_state: Record<string, unknown>;
  behavioral_patterns: Array<Record<string, unknown>>;
  risk_factors: string[];
  strengths: string[];
  recommendations: string[];
  confidence_score: number;
  created_at: string;
  updated_at: string;
}

export interface ValidationReportResponse {
  id: string;
  validation_id: string;
  summary: string;
  detailed_analysis: Record<string, unknown>;
  social_media_insights: Record<string, unknown>;
  timeline_analysis: Record<string, unknown>;
  sentiment_analysis: Record<string, unknown>;
  content_analysis: Record<string, unknown>;
  interaction_patterns: Record<string, unknown>;
  generated_at: string;
}

export interface ValidationDetailResponse {
  validation: ValidationRead;
  profile?: PsychologicalProfileResponse;
  report?: ValidationReportResponse;
  connections: SocialMediaConnectionResponse[];
}

export interface ApiError {
  detail: string;
  status_code: number;
}




