/**
 * API client for psychological profiles endpoints
 */

import { apiClient } from './client';
import type { PsychologicalProfileResponse } from '../types';

export const profilesApi = {
  /**
   * Get profile by validation ID
   */
  getByValidationId: async (validationId: string): Promise<PsychologicalProfileResponse> => {
    const response = await apiClient.get<PsychologicalProfileResponse>(
      `/profile/${validationId}`
    );
    return response.data;
  },
};




