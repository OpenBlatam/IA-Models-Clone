import { apiClient } from '@/lib/api/client';
import type { Template } from '@/types';

export const templateService = {
  async getAll(): Promise<Template[]> {
    return apiClient.getTemplates();
  },

  async getById(id: string): Promise<Template> {
    return apiClient.getTemplate(id);
  },

  async create(template: Omit<Template, 'template_id' | 'created_at' | 'updated_at'>): Promise<Template> {
    return apiClient.createTemplate(template);
  },

  async delete(id: string): Promise<void> {
    return apiClient.deleteTemplate(id);
  },
};



