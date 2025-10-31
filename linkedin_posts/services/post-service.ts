// LinkedIn Posts Service - Functional Components with TypeScript
// Following naming conventions: descriptive variable names, modular architecture

import type {
  LinkedInPost,
  PostContent,
  PostGenerationRequest,
  PostGenerationResponse,
  PostOptimizationResult,
  PostValidationResult,
  PostType,
  PostTone,
  PostStatus,
  EngagementMetrics,
  ContentAnalysisResult
} from '../core/entities';

// Service interfaces for dependency injection
export interface PostRepository {
  readonly createPost: (post: LinkedInPost) => Promise<LinkedInPost>;
  readonly updatePost: (postId: string, updates: Partial<LinkedInPost>) => Promise<LinkedInPost>;
  readonly getPost: (postId: string) => Promise<LinkedInPost | null>;
  readonly listPosts: (userId: string, filters?: PostFilters) => Promise<LinkedInPost[]>;
  readonly deletePost: (postId: string) => Promise<boolean>;
}

export interface PostFilters {
  readonly status?: PostStatus;
  readonly postType?: PostType;
  readonly tone?: PostTone;
  readonly dateRange?: {
    readonly startDate: Date;
    readonly endDate: Date;
  };
  readonly limit?: number;
  readonly offset?: number;
}

export interface AIService {
  readonly analyzeContent: (content: string) => Promise<ContentAnalysisResult>;
  readonly generatePost: (request: PostGenerationRequest) => Promise<PostGenerationResponse>;
  readonly optimizePost: (post: LinkedInPost) => Promise<PostOptimizationResult>;
}

export interface CacheService {
  readonly get: <T>(key: string) => Promise<T | null>;
  readonly set: <T>(key: string, value: T, ttl?: number) => Promise<void>;
  readonly delete: (key: string) => Promise<void>;
  readonly clear: () => Promise<void>;
}

// Main Post Service - Functional Component
export class PostService {
  constructor(
    private readonly postRepository: PostRepository,
    private readonly aiService: AIService,
    private readonly cacheService: CacheService
  ) {}

  // Create new post with validation and AI optimization
  async createPost(request: PostGenerationRequest): Promise<LinkedInPost> {
    const isRequestValid = await this.validatePostRequest(request);
    
    if (!isRequestValid.isValid) {
      throw new Error(`Invalid post request: ${isRequestValid.errors.join(', ')}`);
    }

    // Check cache first
    const cacheKey = this.generateCacheKey(request);
    const cachedPost = await this.cacheService.get<LinkedInPost>(cacheKey);
    
    if (cachedPost) {
      return cachedPost;
    }

    // Generate post using AI
    const generatedResponse = await this.aiService.generatePost(request);
    
    // Create post entity
    const newPost: LinkedInPost = {
      id: this.generatePostId(),
      userId: request.userId || 'default-user',
      title: request.topic,
      content: generatedResponse.post.content,
      postType: request.postType,
      tone: request.tone,
      status: PostStatus.DRAFT,
      createdAt: new Date(),
      updatedAt: new Date(),
      engagement: this.createEmptyEngagementMetrics(),
      aiScore: generatedResponse.aiScore,
      optimizationSuggestions: generatedResponse.optimizationSuggestions,
      keywords: request.keywords || [],
      externalMetadata: {},
      performanceScore: 0,
      reachScore: 0,
      engagementScore: 0
    };

    // Save to repository
    const savedPost = await this.postRepository.createPost(newPost);
    
    // Cache the result
    await this.cacheService.set(cacheKey, savedPost, 3600); // 1 hour TTL
    
    return savedPost;
  }

  // Update existing post
  async updatePost(postId: string, updates: Partial<LinkedInPost>): Promise<LinkedInPost> {
    const existingPost = await this.postRepository.getPost(postId);
    
    if (!existingPost) {
      throw new Error(`Post with ID ${postId} not found`);
    }

    const updatedPost: LinkedInPost = {
      ...existingPost,
      ...updates,
      updatedAt: new Date()
    };

    const savedPost = await this.postRepository.updatePost(postId, updatedPost);
    
    // Clear cache for this post
    await this.cacheService.delete(`post:${postId}`);
    
    return savedPost;
  }

  // Get post by ID with caching
  async getPost(postId: string): Promise<LinkedInPost | null> {
    const cacheKey = `post:${postId}`;
    const cachedPost = await this.cacheService.get<LinkedInPost>(cacheKey);
    
    if (cachedPost) {
      return cachedPost;
    }

    const post = await this.postRepository.getPost(postId);
    
    if (post) {
      await this.cacheService.set(cacheKey, post, 1800); // 30 minutes TTL
    }
    
    return post;
  }

  // List posts with filtering and pagination
  async listPosts(userId: string, filters?: PostFilters): Promise<LinkedInPost[]> {
    const cacheKey = this.generateListCacheKey(userId, filters);
    const cachedPosts = await this.cacheService.get<LinkedInPost[]>(cacheKey);
    
    if (cachedPosts) {
      return cachedPosts;
    }

    const posts = await this.postRepository.listPosts(userId, filters);
    
    await this.cacheService.set(cacheKey, posts, 900); // 15 minutes TTL
    
    return posts;
  }

  // Delete post
  async deletePost(postId: string): Promise<boolean> {
    const isDeleted = await this.postRepository.deletePost(postId);
    
    if (isDeleted) {
      // Clear related cache entries
      await this.cacheService.delete(`post:${postId}`);
      await this.clearListCache();
    }
    
    return isDeleted;
  }

  // Optimize existing post using AI
  async optimizePost(postId: string): Promise<PostOptimizationResult> {
    const existingPost = await this.getPost(postId);
    
    if (!existingPost) {
      throw new Error(`Post with ID ${postId} not found`);
    }

    const optimizationResult = await this.aiService.optimizePost(existingPost);
    
    // Update the post with optimizations
    const optimizedPost = await this.updatePost(postId, {
      content: optimizationResult.optimizedPost.content,
      aiScore: optimizationResult.optimizationScore,
      optimizationSuggestions: optimizationResult.suggestions
    });

    return {
      originalPost: existingPost,
      optimizedPost,
      optimizationScore: optimizationResult.optimizationScore,
      suggestions: optimizationResult.suggestions,
      processingTime: optimizationResult.processingTime
    };
  }

  // Validate post request
  async validatePostRequest(request: PostGenerationRequest): Promise<PostValidationResult> {
    const errors: string[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    // Required field validation
    if (!request.topic || request.topic.trim().length === 0) {
      errors.push('Topic is required');
    }

    if (!request.keyPoints || request.keyPoints.length === 0) {
      errors.push('At least one key point is required');
    }

    if (!request.targetAudience || request.targetAudience.trim().length === 0) {
      errors.push('Target audience is required');
    }

    if (!request.industry || request.industry.trim().length === 0) {
      errors.push('Industry is required');
    }

    // Content length validation
    if (request.topic.length > 200) {
      warnings.push('Topic is quite long, consider making it more concise');
    }

    if (request.keyPoints.some(point => point.length > 500)) {
      warnings.push('Some key points are very long, consider breaking them down');
    }

    // Optimization suggestions
    if (request.keyPoints.length < 2) {
      suggestions.push('Consider adding more key points for better engagement');
    }

    if (!request.keywords || request.keywords.length === 0) {
      suggestions.push('Adding keywords can improve discoverability');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      suggestions
    };
  }

  // Generate analytics for a post
  async generatePostAnalytics(postId: string): Promise<ContentAnalysisResult> {
    const post = await this.getPost(postId);
    
    if (!post) {
      throw new Error(`Post with ID ${postId} not found`);
    }

    return await this.aiService.analyzeContent(post.content.text);
  }

  // Private helper methods
  private generatePostId(): string {
    return `post_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateCacheKey(request: PostGenerationRequest): string {
    const keyData = {
      topic: request.topic,
      targetAudience: request.targetAudience,
      industry: request.industry,
      tone: request.tone,
      postType: request.postType,
      keywords: request.keywords?.sort().join(',')
    };
    
    return `post_gen:${JSON.stringify(keyData)}`;
  }

  private generateListCacheKey(userId: string, filters?: PostFilters): string {
    const filterData = filters ? JSON.stringify(filters) : 'no-filters';
    return `posts_list:${userId}:${filterData}`;
  }

  private createEmptyEngagementMetrics(): EngagementMetrics {
    return {
      likes: 0,
      comments: 0,
      shares: 0,
      clicks: 0,
      impressions: 0,
      reach: 0,
      engagementRate: 0
    };
  }

  private async clearListCache(): Promise<void> {
    // Clear all list cache entries
    // In a real implementation, you might want to be more specific
    await this.cacheService.clear();
  }
}

// Export the service class and interfaces
export type { PostRepository, PostFilters, AIService, CacheService }; 