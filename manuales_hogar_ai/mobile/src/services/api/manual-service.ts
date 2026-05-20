/**
 * Manual Service
 * ==============
 * Service for manual generation and management
 */

import { apiClient } from './api-client';
import type {
  ManualTextRequest,
  ManualResponse,
  Manual,
  ManualListResponse,
  CategoryListResponse,
  ModelListResponse,
  StatisticsResponse,
  HealthResponse,
} from '@/types/api';

export interface GetManualsParams {
  category?: string;
  limit?: number;
  offset?: number;
  search?: string;
}

export interface GetRecentManualsParams {
  limit?: number;
}

export interface GetStatisticsParams {
  days?: number;
}

class ManualService {
  /**
   * Health check
   */
  async healthCheck(): Promise<HealthResponse> {
    return apiClient.get<HealthResponse>('/api/v1/health');
  }

  /**
   * Get available AI models
   */
  async getModels(): Promise<ModelListResponse> {
    return apiClient.get<ModelListResponse>('/api/v1/models');
  }

  /**
   * Get supported categories
   */
  async getCategories(): Promise<CategoryListResponse> {
    return apiClient.get<CategoryListResponse>('/api/v1/categories');
  }

  /**
   * Generate manual from text
   */
  async generateFromText(request: ManualTextRequest): Promise<ManualResponse> {
    return apiClient.post<ManualResponse>('/api/v1/generate-from-text', request);
  }

  /**
   * Generate manual from image
   */
  async generateFromImage(
    imageUri: string,
    problemDescription?: string,
    category?: string
  ): Promise<ManualResponse> {
    const formData = new FormData();
    
    // Convert image URI to blob
    const response = await fetch(imageUri);
    const blob = await response.blob();
    const filename = imageUri.split('/').pop() || 'image.jpg';
    
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: filename,
    } as any);

    if (problemDescription) {
      formData.append('problem_description', problemDescription);
    }
    if (category) {
      formData.append('category', category);
    }

    return apiClient.postFormData<ManualResponse>('/api/v1/generate-from-image', formData);
  }

  /**
   * Generate manual from multiple images
   */
  async generateFromMultipleImages(
    imageUris: string[],
    problemDescription?: string,
    category?: string
  ): Promise<ManualResponse> {
    const formData = new FormData();

    for (let i = 0; i < imageUris.length; i++) {
      const uri = imageUris[i];
      const filename = uri.split('/').pop() || `image-${i}.jpg`;
      
      formData.append('files', {
        uri,
        type: 'image/jpeg',
        name: filename,
      } as any);
    }

    if (problemDescription) {
      formData.append('problem_description', problemDescription);
    }
    if (category) {
      formData.append('category', category);
    }

    return apiClient.postFormData<ManualResponse>('/api/v1/generate-from-multiple-images', formData);
  }

  /**
   * Generate combined manual (text + image)
   */
  async generateCombined(
    problemDescription: string,
    imageUri?: string,
    category?: string
  ): Promise<ManualResponse> {
    const formData = new FormData();

    formData.append('problem_description', problemDescription);

    if (imageUri) {
      const filename = imageUri.split('/').pop() || 'image.jpg';
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: filename,
      } as any);
    }

    if (category) {
      formData.append('category', category);
    }

    return apiClient.postFormData<ManualResponse>('/api/v1/generate-combined', formData);
  }

  /**
   * Get manuals list
   */
  async getManuals(params?: GetManualsParams): Promise<ManualListResponse> {
    const queryParams = new URLSearchParams();
    
    if (params?.category) queryParams.append('category', params.category);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    if (params?.search) queryParams.append('search', params.search);

    const queryString = queryParams.toString();
    const url = `/api/v1/manuals${queryString ? `?${queryString}` : ''}`;

    return apiClient.get<ManualListResponse>(url);
  }

  /**
   * Get recent manuals
   */
  async getRecentManuals(params?: GetRecentManualsParams): Promise<ManualListResponse> {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    const queryString = queryParams.toString();
    const url = `/api/v1/manuals/recent${queryString ? `?${queryString}` : ''}`;

    return apiClient.get<ManualListResponse>(url);
  }

  /**
   * Get manual by ID
   */
  async getManualById(id: string): Promise<Manual> {
    return apiClient.get<Manual>(`/api/v1/manuals/${id}`);
  }

  /**
   * Get statistics
   */
  async getStatistics(params?: GetStatisticsParams): Promise<StatisticsResponse> {
    const queryParams = new URLSearchParams();
    if (params?.days) queryParams.append('days', params.days.toString());

    const queryString = queryParams.toString();
    const url = `/api/v1/statistics${queryString ? `?${queryString}` : ''}`;

    return apiClient.get<StatisticsResponse>(url);
  }
}

export const manualService = new ManualService();




