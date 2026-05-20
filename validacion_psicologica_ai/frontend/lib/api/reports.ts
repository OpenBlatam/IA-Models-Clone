/**
 * API client for validation reports endpoints
 */

import { apiClient } from './client';
import type { ValidationReportResponse } from '../types';

export const reportsApi = {
  /**
   * Get report by validation ID
   */
  getByValidationId: async (validationId: string): Promise<ValidationReportResponse> => {
    const response = await apiClient.get<ValidationReportResponse>(
      `/report/${validationId}`
    );
    return response.data;
  },

  /**
   * Export report
   */
  export: async (validationId: string, format: 'json' | 'pdf' | 'html'): Promise<Blob> => {
    const response = await apiClient.get(`/validations/${validationId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },
};




