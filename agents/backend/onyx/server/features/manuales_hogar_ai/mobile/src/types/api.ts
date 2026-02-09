/**
 * API Types
 * =========
 * TypeScript interfaces matching the backend API models
 */

export interface ManualTextRequest {
  problem_description: string;
  category?: string;
  model?: string;
  include_safety?: boolean;
  include_tools?: boolean;
  include_materials?: boolean;
}

export interface ManualResponse {
  success: boolean;
  manual?: string;
  category?: string;
  model_used?: string;
  tokens_used?: number;
  format: string;
  image_analysis?: string;
  detected_category?: string;
  images_count?: number;
  error?: string;
}

export interface HealthResponse {
  status: string;
  healthy: boolean;
  timestamp?: string;
  error?: string;
}

export interface Manual {
  id: string;
  manual: string;
  category: string;
  model_used?: string;
  tokens_used?: number;
  created_at: string;
  updated_at?: string;
  problem_description?: string;
  images_count?: number;
}

export interface ManualListResponse {
  success: boolean;
  manuals: Manual[];
  total: number;
  limit: number;
  offset: number;
}

export interface Category {
  id: string;
  name: string;
  display_name: string;
  icon?: string;
  description?: string;
}

export interface CategoryListResponse {
  success: boolean;
  categories: Category[];
  total: number;
}

export interface Model {
  id: string;
  name: string;
  description?: string;
  context_length?: number;
  pricing?: {
    prompt?: string;
    completion?: string;
  };
}

export interface ModelListResponse {
  success: boolean;
  models: Model[];
  total: number;
}

export interface StatisticsResponse {
  success: boolean;
  total_manuals: number;
  total_tokens: number;
  categories_count: Record<string, number>;
  models_used: Record<string, number>;
  period_days: number;
}

export interface CacheStatsResponse {
  success: boolean;
  memory_cache: {
    hits: number;
    misses: number;
    size: number;
  };
  db_cache?: {
    hits: number;
    misses: number;
    size: number;
    expired: number;
  };
}




