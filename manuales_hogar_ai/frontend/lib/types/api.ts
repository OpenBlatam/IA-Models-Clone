export type Category =
  | 'plomeria'
  | 'techos'
  | 'carpinteria'
  | 'electricidad'
  | 'albanileria'
  | 'pintura'
  | 'herreria'
  | 'jardineria'
  | 'general';

export interface CategoryInfo {
  plomeria: string;
  techos: string;
  carpinteria: string;
  electricidad: string;
  albanileria: string;
  pintura: string;
  herreria: string;
  jardineria: string;
  general: string;
}

export interface ManualTextRequest {
  problem_description: string;
  category?: Category;
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

export interface Model {
  id: string;
  name?: string;
  description?: string;
  context_length?: number;
  pricing?: {
    prompt?: string;
    completion?: string;
  };
}

export interface ModelsResponse {
  success: boolean;
  models: Model[];
  total: number;
}

export interface CategoriesResponse {
  success: boolean;
  categories: Category[];
  category_names: CategoryInfo;
}

export interface ManualListItem {
  id: number;
  problem_description: string;
  category: string;
  model_used?: string;
  tokens_used: number;
  images_count: number;
  created_at: string;
}

export interface ManualDetailResponse {
  id: number;
  problem_description: string;
  category: string;
  manual_content: string;
  model_used?: string;
  tokens_used: number;
  image_analysis?: string;
  detected_category?: string;
  images_count: number;
  format: string;
  created_at: string;
  updated_at?: string;
}

export interface StatisticsResponse {
  total_manuals: number;
  category_stats: Record<string, number>;
  total_tokens: number;
  top_models: Array<{
    model: string;
    count: number;
  }>;
  period_days: number;
}

export interface RatingRequest {
  rating: number;
  comment?: string;
}

export interface RatingResponse {
  id: number;
  manual_id: number;
  user_id?: string;
  rating: number;
  comment?: string;
  created_at: string;
}

export interface FavoriteResponse {
  id: number;
  manual_id: number;
  user_id: string;
  created_at: string;
}

export interface AdvancedSearchRequest {
  query?: string;
  category?: Category;
  difficulty?: string;
  min_rating?: number;
  max_rating?: number;
  tags?: string[];
  date_from?: string;
  date_to?: string;
  limit?: number;
  offset?: number;
}

export interface SearchResponse {
  success: boolean;
  query?: string;
  results: ManualListItem[];
  total: number;
  limit?: number;
  offset?: number;
}

export interface SemanticSearchResponse {
  success: boolean;
  query: string;
  results: Array<{
    manual: ManualListItem;
    similarity: number;
    score: number;
  }>;
  total: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  hit_rate: number;
}

export interface CacheStatsResponse {
  success: boolean;
  cache_stats: CacheStats;
}

