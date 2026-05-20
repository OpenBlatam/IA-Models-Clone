import { apiClient } from '@/utils/api-client';
import { API_ENDPOINTS } from '@/utils/config';
import type { Template, CustomTemplate, VideoGenerationRequest } from '@/types/api';

export const templateService = {
  async listTemplates(): Promise<{ templates: Template[] }> {
    return apiClient.get<{ templates: Template[] }>(API_ENDPOINTS.TEMPLATES.LIST);
  },

  async getTemplate(name: string): Promise<Template> {
    return apiClient.get<Template>(API_ENDPOINTS.TEMPLATES.GET(name));
  },

  async generateFromTemplate(
    templateName: string,
    scriptText: string,
    language = 'es'
  ): Promise<unknown> {
    return apiClient.post<unknown>(
      API_ENDPOINTS.TEMPLATES.GENERATE(templateName),
      { script_text: scriptText, language }
    );
  },

  async createCustomTemplate(
    name: string,
    description: string,
    config: {
      video_config: unknown;
      audio_config: unknown;
      subtitle_config: unknown;
    },
    isPublic = false
  ): Promise<CustomTemplate> {
    return apiClient.post<CustomTemplate>(API_ENDPOINTS.CUSTOM_TEMPLATES.CREATE, {
      name,
      description,
      config,
      is_public: isPublic,
    });
  },

  async listCustomTemplates(userOnly = false): Promise<{ templates: CustomTemplate[] }> {
    const params = userOnly ? '?user_only=true' : '';
    return apiClient.get<{ templates: CustomTemplate[] }>(
      `${API_ENDPOINTS.CUSTOM_TEMPLATES.LIST}${params}`
    );
  },
};


