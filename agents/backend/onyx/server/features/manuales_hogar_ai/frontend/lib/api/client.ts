import axios, { AxiosInstance, AxiosError } from 'axios';
import { API_BASE_URL, API_ENDPOINTS } from './config';
import { buildManualFormData } from './form-data-builder';
import { buildQueryString } from './query-builder';
import { MESSAGES } from '../constants';
import type {
  ManualTextRequest,
  ManualResponse,
  HealthResponse,
  ModelsResponse,
  CategoriesResponse,
  ManualListItem,
  ManualDetailResponse,
  StatisticsResponse,
  RatingRequest,
  RatingResponse,
  FavoriteResponse,
  AdvancedSearchRequest,
  SearchResponse,
  SemanticSearchResponse,
  CacheStatsResponse,
  Category,
} from '../types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 120000,
    });

    this.setupInterceptors();
  }

  private setupInterceptors = (): void => {
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          const message =
            (error.response.data as { detail?: string })?.detail ||
            error.message;
          throw new Error(message);
        }
        throw error;
      }
    );
  };

  // Health Check
  getHealth = async (): Promise<HealthResponse> => {
    const response = await this.client.get<HealthResponse>(
      API_ENDPOINTS.health
    );
    return response.data;
  };

  // Models
  getModels = async (): Promise<ModelsResponse> => {
    const response = await this.client.get<ModelsResponse>(
      API_ENDPOINTS.models
    );
    return response.data;
  };

  // Categories
  getCategories = async (): Promise<CategoriesResponse> => {
    const response = await this.client.get<CategoriesResponse>(
      API_ENDPOINTS.categories
    );
    return response.data;
  };

  // Generate Manual from Text
  generateFromText = async (
    request: ManualTextRequest
  ): Promise<ManualResponse> => {
    const response = await this.client.post<ManualResponse>(
      API_ENDPOINTS.generateFromText,
      request
    );
    return response.data;
  };

  // Generate Manual from Image
  generateFromImage = async (
    file: File,
    problemDescription?: string,
    category?: Category,
    model?: string,
    includeSafety = true,
    includeTools = true,
    includeMaterials = true
  ): Promise<ManualResponse> => {
    const formData = buildManualFormData(
      { problemDescription, category, model, includeSafety, includeTools, includeMaterials },
      file
    );

    const response = await this.client.post<ManualResponse>(
      API_ENDPOINTS.generateFromImage,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  };

  // Generate Manual Combined
  generateCombined = async (
    problemDescription: string,
    file?: File,
    category: Category = 'general',
    model?: string,
    includeSafety = true,
    includeTools = true,
    includeMaterials = true
  ): Promise<ManualResponse> => {
    const formData = buildManualFormData(
      { problemDescription, category, model, includeSafety, includeTools, includeMaterials },
      file
    );

    const response = await this.client.post<ManualResponse>(
      API_ENDPOINTS.generateCombined,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  };

  // Generate Manual from Multiple Images
  generateFromMultipleImages = async (
    files: File[],
    problemDescription?: string,
    category?: Category,
    model?: string,
    includeSafety = true,
    includeTools = true,
    includeMaterials = true
  ): Promise<ManualResponse> => {
    if (files.length === 0 || files.length > 5) {
      throw new Error(MESSAGES.FILE.INVALID_COUNT);
    }

    const formData = buildManualFormData(
      { problemDescription, category, model, includeSafety, includeTools, includeMaterials },
      files
    );

    const response = await this.client.post<ManualResponse>(
      API_ENDPOINTS.generateFromMultipleImages,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  };

  // Manuals List
  getManuals = async (
    category?: Category,
    search?: string,
    limit = 20,
    offset = 0
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ category, search, limit, offset });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.manuals}?${queryString}`
    );
    return response.data;
  };

  // Manual Detail
  getManual = async (id: number): Promise<ManualDetailResponse> => {
    const response = await this.client.get<ManualDetailResponse>(
      API_ENDPOINTS.manualDetail(id)
    );
    return response.data;
  };

  // Recent Manuals
  getRecentManuals = async (
    limit = 10,
    category?: Category
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ limit, category });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.recentManuals}?${queryString}`
    );
    return response.data;
  };

  // Statistics
  getStatistics = async (days = 30): Promise<StatisticsResponse> => {
    const queryString = buildQueryString({ days });
    const response = await this.client.get<StatisticsResponse>(
      `${API_ENDPOINTS.statistics}?${queryString}`
    );
    return response.data;
  };

  // Search
  search = async (
    query: string,
    category?: Category,
    limit = 20,
    offset = 0
  ): Promise<SearchResponse> => {
    const queryString = buildQueryString({ q: query, category, limit, offset });
    const response = await this.client.get<SearchResponse>(
      `${API_ENDPOINTS.search}?${queryString}`
    );
    return response.data;
  };

  // Semantic Search
  semanticSearch = async (
    query: string,
    category?: Category,
    limit = 10,
    threshold = 0.5
  ): Promise<SemanticSearchResponse> => {
    const queryString = buildQueryString({ query, category, limit, threshold });
    const response = await this.client.post<SemanticSearchResponse>(
      `${API_ENDPOINTS.semanticSearch}?${queryString}`
    );
    return response.data;
  };

  // Advanced Search
  advancedSearch = async (
    request: AdvancedSearchRequest
  ): Promise<SearchResponse> => {
    const response = await this.client.post<SearchResponse>(
      API_ENDPOINTS.advancedSearch,
      request
    );
    return response.data;
  };

  // Search Suggestions
  getSearchSuggestions = async (
    query: string,
    limit = 10
  ): Promise<{ success: boolean; query: string; suggestions: string[] }> => {
    const queryString = buildQueryString({ q: query, limit });
    const response = await this.client.get<{
      success: boolean;
      query: string;
      suggestions: string[];
    }>(`${API_ENDPOINTS.searchSuggestions}?${queryString}`);
    return response.data;
  };

  // Ratings
  addRating = async (
    manualId: number,
    request: RatingRequest,
    userId?: string
  ): Promise<RatingResponse> => {
    const queryString = userId ? buildQueryString({ user_id: userId }) : '';
    const response = await this.client.post<RatingResponse>(
      `${API_ENDPOINTS.rating(manualId)}${queryString ? `?${queryString}` : ''}`,
      request
    );
    return response.data;
  };

  getRatings = async (
    manualId: number,
    limit = 20,
    offset = 0
  ): Promise<RatingResponse[]> => {
    const queryString = buildQueryString({ limit, offset });
    const response = await this.client.get<RatingResponse[]>(
      `${API_ENDPOINTS.ratings(manualId)}?${queryString}`
    );
    return response.data;
  };

  // Favorites
  addFavorite = async (
    manualId: number,
    userId: string
  ): Promise<FavoriteResponse> => {
    const queryString = buildQueryString({ user_id: userId });
    const response = await this.client.post<FavoriteResponse>(
      `${API_ENDPOINTS.favorite(manualId)}?${queryString}`
    );
    return response.data;
  };

  removeFavorite = async (
    manualId: number,
    userId: string
  ): Promise<{ success: boolean; message: string }> => {
    const queryString = buildQueryString({ user_id: userId });
    const response = await this.client.delete<{
      success: boolean;
      message: string;
    }>(`${API_ENDPOINTS.favorite(manualId)}?${queryString}`);
    return response.data;
  };

  checkFavorite = async (
    manualId: number,
    userId: string
  ): Promise<{ is_favorite: boolean; manual_id: number; user_id: string }> => {
    const queryString = buildQueryString({ user_id: userId });
    const response = await this.client.get<{
      is_favorite: boolean;
      manual_id: number;
      user_id: string;
    }>(`${API_ENDPOINTS.checkFavorite(manualId)}?${queryString}`);
    return response.data;
  };

  getUserFavorites = async (
    userId: string,
    limit = 20,
    offset = 0
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ limit, offset });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.userFavorites(userId)}?${queryString}`
    );
    return response.data;
  };

  // Recommendations
  getPopularManuals = async (
    category?: Category,
    limit = 10
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ category, limit });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.popularManuals}?${queryString}`
    );
    return response.data;
  };

  getTopRatedManuals = async (
    category?: Category,
    limit = 10,
    minRatings = 3
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ category, limit, min_ratings: minRatings });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.topRatedManuals}?${queryString}`
    );
    return response.data;
  };

  getSimilarManuals = async (
    manualId: number,
    limit = 5
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ limit });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.similarManuals(manualId)}?${queryString}`
    );
    return response.data;
  };

  getTrendingManuals = async (
    days = 7,
    limit = 10
  ): Promise<ManualListItem[]> => {
    const queryString = buildQueryString({ days, limit });
    const response = await this.client.get<ManualListItem[]>(
      `${API_ENDPOINTS.trendingManuals}?${queryString}`
    );
    return response.data;
  };

  // Cache
  getCacheStats = async (): Promise<CacheStatsResponse> => {
    const response = await this.client.get<CacheStatsResponse>(
      API_ENDPOINTS.cacheStats
    );
    return response.data;
  };

  clearCache = async (): Promise<{ success: boolean; message: string }> => {
    const response = await this.client.delete<{
      success: boolean;
      message: string;
    }>(API_ENDPOINTS.clearCache);
    return response.data;
  };

  // Analytics
  getComprehensiveStats = async (days = 30): Promise<{ success: boolean; stats: any }> => {
    const queryString = buildQueryString({ days });
    const response = await this.client.get<{ success: boolean; stats: any }>(
      `${API_ENDPOINTS.comprehensiveStats}?${queryString}`
    );
    return response.data;
  };

  getTrends = async (days = 30, interval: 'day' | 'week' = 'day'): Promise<{
    success: boolean;
    trends: any;
    period_days: number;
    interval: string;
  }> => {
    const queryString = buildQueryString({ days, interval });
    const response = await this.client.get<{
      success: boolean;
      trends: any;
      period_days: number;
      interval: string;
    }>(`${API_ENDPOINTS.trends}?${queryString}`);
    return response.data;
  };

  // Export
  exportToMarkdown = async (manualId: number): Promise<Blob> => {
    const response = await this.client.get(API_ENDPOINTS.exportMarkdown(manualId), {
      responseType: 'blob',
    });
    return response.data;
  };

  exportToText = async (manualId: number): Promise<Blob> => {
    const response = await this.client.get(API_ENDPOINTS.exportText(manualId), {
      responseType: 'blob',
    });
    return response.data;
  };

  exportToJson = async (manualId: number): Promise<Blob> => {
    const response = await this.client.get(API_ENDPOINTS.exportJson(manualId), {
      responseType: 'blob',
    });
    return response.data;
  };
}

export const apiClient = new ApiClient();

