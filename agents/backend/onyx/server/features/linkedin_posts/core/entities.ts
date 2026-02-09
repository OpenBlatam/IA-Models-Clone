// LinkedIn Posts Core Entities - TypeScript with Strict Mode
// Following naming conventions: lowercase with dashes, descriptive variable names

export interface LinkedInPost {
  readonly id: string;
  readonly userId: string;
  readonly title: string;
  readonly content: PostContent;
  readonly postType: PostType;
  readonly tone: PostTone;
  readonly status: PostStatus;
  readonly createdAt: Date;
  readonly updatedAt: Date;
  readonly scheduledAt?: Date;
  readonly publishedAt?: Date;
  readonly engagement: EngagementMetrics;
  readonly aiScore: number;
  readonly optimizationSuggestions: string[];
  readonly keywords: string[];
  readonly linkedinPostId?: string;
  readonly externalMetadata: Record<string, unknown>;
  readonly performanceScore: number;
  readonly reachScore: number;
  readonly engagementScore: number;
}

export interface PostContent {
  readonly text: string;
  readonly hashtags: string[];
  readonly mentions: string[];
  readonly links: string[];
  readonly images: string[];
  readonly callToAction?: string;
}

export interface EngagementMetrics {
  readonly likes: number;
  readonly comments: number;
  readonly shares: number;
  readonly clicks: number;
  readonly impressions: number;
  readonly reach: number;
  readonly engagementRate: number;
}

// Use maps instead of enums for better type safety
export const PostType = {
  TEXT: 'text',
  IMAGE: 'image',
  VIDEO: 'video',
  ARTICLE: 'article',
  POLL: 'poll',
  EVENT: 'event'
} as const;

export type PostType = typeof PostType[keyof typeof PostType];

export const PostTone = {
  PROFESSIONAL: 'professional',
  CASUAL: 'casual',
  FRIENDLY: 'friendly',
  AUTHORITATIVE: 'authoritative',
  INSPIRATIONAL: 'inspirational'
} as const;

export type PostTone = typeof PostTone[keyof typeof PostTone];

export const PostStatus = {
  DRAFT: 'draft',
  SCHEDULED: 'scheduled',
  PUBLISHED: 'published',
  ARCHIVED: 'archived',
  DELETED: 'deleted'
} as const;

export type PostStatus = typeof PostStatus[keyof typeof PostStatus];

// Functional components with TypeScript interfaces
export interface PostGeneratorConfig {
  readonly targetAudience: string;
  readonly industry: string;
  readonly tone: PostTone;
  readonly postType: PostType;
  readonly keywords?: string[];
  readonly additionalContext?: string;
}

export interface PostOptimizationResult {
  readonly originalPost: LinkedInPost;
  readonly optimizedPost: LinkedInPost;
  readonly optimizationScore: number;
  readonly suggestions: string[];
  readonly processingTime: number;
}

export interface PostAnalytics {
  readonly postId: string;
  readonly engagementMetrics: EngagementMetrics;
  readonly performanceScore: number;
  readonly reachScore: number;
  readonly engagementScore: number;
  readonly trendingKeywords: string[];
  readonly audienceInsights: AudienceInsights;
}

export interface AudienceInsights {
  readonly demographics: Demographics;
  readonly interests: string[];
  readonly engagementPatterns: EngagementPattern[];
  readonly optimalPostingTimes: string[];
}

export interface Demographics {
  readonly ageRanges: Record<string, number>;
  readonly locations: Record<string, number>;
  readonly industries: Record<string, number>;
  readonly jobTitles: Record<string, number>;
}

export interface EngagementPattern {
  readonly dayOfWeek: string;
  readonly timeOfDay: string;
  readonly engagementRate: number;
  readonly postType: PostType;
}

// Utility types for better type safety
export type PostId = string;
export type UserId = string;
export type Hashtag = string;
export type Keyword = string;

// Strict validation interfaces
export interface PostValidationResult {
  readonly isValid: boolean;
  readonly errors: string[];
  readonly warnings: string[];
  readonly suggestions: string[];
}

export interface PostGenerationRequest {
  readonly topic: string;
  readonly keyPoints: string[];
  readonly targetAudience: string;
  readonly industry: string;
  readonly tone: PostTone;
  readonly postType: PostType;
  readonly keywords?: Keyword[];
  readonly additionalContext?: string;
}

export interface PostGenerationResponse {
  readonly post: LinkedInPost;
  readonly generationTime: number;
  readonly aiScore: number;
  readonly optimizationSuggestions: string[];
}

// Cache interfaces for performance optimization
export interface CacheConfig {
  readonly maxSize: number;
  readonly ttl: number;
  readonly enableCompression: boolean;
  readonly enablePredictiveCaching: boolean;
}

export interface CacheMetrics {
  readonly hitRate: number;
  readonly missRate: number;
  readonly averageResponseTime: number;
  readonly totalRequests: number;
  readonly cacheSize: number;
}

// AI/ML interfaces
export interface AIModelConfig {
  readonly modelName: string;
  readonly enableGPU: boolean;
  readonly enableQuantization: boolean;
  readonly batchSize: number;
  readonly maxConcurrentRequests: number;
}

export interface ContentAnalysisResult {
  readonly sentimentScore: number;
  readonly readabilityScore: number;
  readonly engagementScore: number;
  readonly keywordDensity: number;
  readonly structureScore: number;
  readonly callToActionScore: number;
}

// Performance monitoring interfaces
export interface PerformanceMetrics {
  readonly responseTime: number;
  readonly throughput: number;
  readonly errorRate: number;
  readonly cpuUsage: number;
  readonly memoryUsage: number;
  readonly cacheHitRate: number;
}

export interface SystemHealth {
  readonly status: 'healthy' | 'warning' | 'critical';
  readonly timestamp: Date;
  readonly metrics: PerformanceMetrics;
  readonly alerts: Alert[];
}

export interface Alert {
  readonly type: 'performance' | 'error' | 'warning';
  readonly message: string;
  readonly severity: 'low' | 'medium' | 'high' | 'critical';
  readonly timestamp: Date;
}

// All interfaces are already exported above 