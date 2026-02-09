import { apiClient } from '@/utils/api-client';
import { API_ENDPOINTS } from '@/utils/config';
import type { Analytics, Recommendations, QuotaInfo } from '@/types/api';

export const analyticsService = {
  async getAnalytics(): Promise<Analytics> {
    return apiClient.get<Analytics>(API_ENDPOINTS.ANALYTICS);
  },

  async getRecommendations(
    scriptText: string,
    language = 'es',
    platform?: string,
    contentType = 'general'
  ): Promise<Recommendations> {
    const params = new URLSearchParams({
      script_text: scriptText,
      language,
      content_type: contentType,
    });
    if (platform) {
      params.append('platform', platform);
    }
    return apiClient.get<Recommendations>(
      `${API_ENDPOINTS.RECOMMENDATIONS}?${params.toString()}`
    );
  },

  async getQuota(): Promise<QuotaInfo> {
    return apiClient.get<QuotaInfo>(API_ENDPOINTS.QUOTA);
  },
};


