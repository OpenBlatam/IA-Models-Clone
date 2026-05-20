/**
 * Templates API
 * Handles all template-related API operations
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { Template, TemplateCreate } from '@/types';

/**
 * Get all templates
 * @returns Array of templates
 */
export const getAllTemplates = async (): Promise<Template[]> => {
  return apiGet<Template[]>(API_ENDPOINTS.TEMPLATES);
};

/**
 * Get a single template by ID
 * @param templateId - The template ID
 * @returns Template data
 */
export const getTemplateById = async (templateId: string): Promise<Template> => {
  return apiGet<Template>(`${API_ENDPOINTS.TEMPLATES}/${templateId}`);
};

/**
 * Create a new template
 * @param template - Template creation data
 * @returns Created template
 */
export const createTemplate = async (template: TemplateCreate): Promise<Template> => {
  return apiPost<Template>(API_ENDPOINTS.TEMPLATES, template);
};

/**
 * Update an existing template
 * @param templateId - The template ID
 * @param template - Template update data
 * @returns Updated template
 */
export const updateTemplate = async (
  templateId: string,
  template: TemplateCreate
): Promise<Template> => {
  return apiPut<Template>(`${API_ENDPOINTS.TEMPLATES}/${templateId}`, template);
};

/**
 * Delete a template
 * @param templateId - The template ID
 * @returns Deletion result
 */
export const deleteTemplate = async (templateId: string): Promise<void> => {
  return apiDelete<void>(`${API_ENDPOINTS.TEMPLATES}/${templateId}`);
};

/**
 * Render a template with variables
 * @param templateId - The template ID
 * @param variables - Variables to substitute in the template
 * @returns Rendered template content
 */
export const renderTemplate = async (
  templateId: string,
  variables: Record<string, string>
): Promise<{ content: string }> => {
  return apiPost<{ content: string }>(`${API_ENDPOINTS.TEMPLATES}/${templateId}/render`, {
    variables,
  });
};


