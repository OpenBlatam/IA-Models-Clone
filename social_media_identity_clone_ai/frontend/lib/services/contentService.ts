import { apiClient } from '@/lib/api/client';
import type { GeneratedContent, GenerateContentRequest, GenerateContentResponse } from '@/types';

export const contentService = {
  async generate(request: GenerateContentRequest): Promise<GenerateContentResponse> {
    return apiClient.generateContent(request);
  },

  async getByIdentityId(identityId: string, limit = 10): Promise<GeneratedContent[]> {
    return apiClient.getGeneratedContent(identityId, limit);
  },

  async search(query: string): Promise<GeneratedContent[]> {
    return apiClient.searchContent(query);
  },
};



