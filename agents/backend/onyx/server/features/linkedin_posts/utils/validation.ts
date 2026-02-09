import { z } from 'zod';

// Validation schemas using Zod
export const PostGenerationSchema = z.object({
  topic: z.string()
    .min(1, 'Topic is required')
    .max(200, 'Topic must be less than 200 characters')
    .trim(),
  
  keyPoints: z.array(z.string())
    .min(1, 'At least one key point is required')
    .max(10, 'Maximum 10 key points allowed')
    .refine(points => points.every(point => point.trim().length > 0), {
      message: 'All key points must not be empty'
    }),
  
  targetAudience: z.string()
    .min(1, 'Target audience is required')
    .max(100, 'Target audience must be less than 100 characters')
    .trim(),
  
  industry: z.string()
    .min(1, 'Industry is required')
    .max(50, 'Industry must be less than 50 characters')
    .trim(),
  
  tone: z.enum(['professional', 'casual', 'friendly'], {
    errorMap: () => ({ message: 'Tone must be professional, casual, or friendly' })
  }),
  
  postType: z.enum(['announcement', 'educational', 'update', 'insight'], {
    errorMap: () => ({ message: 'Post type must be announcement, educational, update, or insight' })
  }),
  
  keywords: z.array(z.string())
    .max(10, 'Maximum 10 keywords allowed')
    .optional(),
  
  additionalContext: z.string()
    .max(500, 'Additional context must be less than 500 characters')
    .optional(),
});

export const PostUpdateSchema = z.object({
  id: z.string().min(1, 'Post ID is required'),
  content: z.string()
    .min(1, 'Content is required')
    .max(3000, 'Content must be less than 3000 characters'),
  status: z.enum(['draft', 'published', 'scheduled']).optional(),
  optimizationScore: z.number().min(0).max(1).optional(),
});

// Error types
export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

export interface ApiError {
  message: string;
  code: string;
  statusCode: number;
  details?: Record<string, any>;
}

// Validation functions
export function validatePostGeneration(data: unknown): { success: true; data: z.infer<typeof PostGenerationSchema> } | { success: false; errors: ValidationError[] } {
  try {
    const validatedData = PostGenerationSchema.parse(data);
    return { success: true, data: validatedData };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const errors: ValidationError[] = error.errors.map(err => ({
        field: err.path.join('.'),
        message: err.message,
        code: 'VALIDATION_ERROR',
      }));
      return { success: false, errors };
    }
    return { 
      success: false, 
      errors: [{ field: 'unknown', message: 'Unknown validation error', code: 'UNKNOWN_ERROR' }] 
    };
  }
}

export function validatePostUpdate(data: unknown): { success: true; data: z.infer<typeof PostUpdateSchema> } | { success: false; errors: ValidationError[] } {
  try {
    const validatedData = PostUpdateSchema.parse(data);
    return { success: true, data: validatedData };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const errors: ValidationError[] = error.errors.map(err => ({
        field: err.path.join('.'),
        message: err.message,
        code: 'VALIDATION_ERROR',
      }));
      return { success: false, errors };
    }
    return { 
      success: false, 
      errors: [{ field: 'unknown', message: 'Unknown validation error', code: 'UNKNOWN_ERROR' }] 
    };
  }
}

// Business logic validation
export function validateContentQuality(content: string): { isValid: boolean; suggestions: string[] } {
  const suggestions: string[] = [];
  let isValid = true;

  // Check content length
  if (content.length < 50) {
    suggestions.push('Content is too short. Consider adding more details.');
    isValid = false;
  }

  if (content.length > 3000) {
    suggestions.push('Content is too long. Consider breaking it into multiple posts.');
    isValid = false;
  }

  // Check for hashtags
  const hashtagCount = (content.match(/#/g) || []).length;
  if (hashtagCount < 2) {
    suggestions.push('Consider adding more hashtags for better discoverability.');
  } else if (hashtagCount > 10) {
    suggestions.push('Too many hashtags can look spammy. Consider reducing to 3-5 hashtags.');
  }

  // Check for engagement elements
  if (!content.includes('?') && !content.includes('!')) {
    suggestions.push('Consider adding questions or exclamations to increase engagement.');
  }

  // Check for call-to-action
  const ctaPhrases = ['click', 'learn more', 'read more', 'check out', 'visit', 'sign up', 'join'];
  const hasCTA = ctaPhrases.some(phrase => content.toLowerCase().includes(phrase));
  if (!hasCTA) {
    suggestions.push('Consider adding a call-to-action to encourage engagement.');
  }

  return { isValid, suggestions };
}

// Error handling utilities
export function createApiError(message: string, code: string, statusCode: number = 400, details?: Record<string, any>): ApiError {
  return {
    message,
    code,
    statusCode,
    details,
  };
}

export function handleApiError(error: unknown): ApiError {
  if (error instanceof Error) {
    return createApiError(error.message, 'API_ERROR', 500);
  }
  
  if (typeof error === 'string') {
    return createApiError(error, 'API_ERROR', 500);
  }
  
  return createApiError('An unexpected error occurred', 'UNKNOWN_ERROR', 500);
}

// Rate limiting validation
export function validateRateLimit(userId: string, lastRequestTime: number): { allowed: boolean; waitTime?: number } {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  const minInterval = 1000; // 1 second between requests

  if (timeSinceLastRequest < minInterval) {
    return {
      allowed: false,
      waitTime: minInterval - timeSinceLastRequest,
    };
  }

  return { allowed: true };
}

// Content moderation validation
export function validateContentModeration(content: string): { isAppropriate: boolean; flaggedWords?: string[] } {
  const inappropriateWords = [
    'spam', 'scam', 'fake', 'clickbait', 'urgent', 'limited time',
    'act now', 'don\'t miss out', 'exclusive offer'
  ];

  const flaggedWords = inappropriateWords.filter(word => 
    content.toLowerCase().includes(word.toLowerCase())
  );

  return {
    isAppropriate: flaggedWords.length === 0,
    flaggedWords: flaggedWords.length > 0 ? flaggedWords : undefined,
  };
}

// Export types
export type PostGenerationRequest = z.infer<typeof PostGenerationSchema>;
export type PostUpdateRequest = z.infer<typeof PostUpdateSchema>; 