import { apiClient } from '@/lib/api/client';
import type { ExtractProfileRequest, ExtractProfileResponse } from '@/types';

export const profileService = {
  async extract(request: ExtractProfileRequest): Promise<ExtractProfileResponse> {
    return apiClient.extractProfile(request);
  },
};



