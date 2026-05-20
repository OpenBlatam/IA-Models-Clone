/**
 * API client for validations endpoints
 */

import { apiClient } from './client';
import type {
  ValidationCreate,
  ValidationRead,
  ValidationDetailResponse,
} from '../types';

export const validationsApi = {
  /**
   * Get all validations
   */
  getAll: async (): Promise<ValidationRead[]> => {
    const response = await apiClient.get<ValidationRead[]>('/validations');
    return response.data;
  },

  /**
   * Get validation by ID
   */
  getById: async (id: string): Promise<ValidationDetailResponse> => {
    const response = await apiClient.get<ValidationDetailResponse>(`/validations/${id}`);
    return response.data;
  },

  /**
   * Create new validation
   */
  create: async (data: ValidationCreate): Promise<ValidationRead> => {
    const response = await apiClient.post<ValidationRead>('/validations', data);
    return response.data;
  },

  /**
   * Run validation analysis
   */
  run: async (id: string): Promise<ValidationRead> => {
    const response = await apiClient.post<ValidationRead>(`/validations/${id}/run`);
    return response.data;
  },
};




