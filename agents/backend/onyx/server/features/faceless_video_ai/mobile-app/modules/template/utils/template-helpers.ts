/**
 * Template helper utilities
 */

import type { Template, CustomTemplate } from '@/types/api';

export function isCustomTemplate(
  template: Template | CustomTemplate
): template is CustomTemplate {
  return 'template_id' in template && 'user_id' in template;
}

export function getTemplateName(template: Template | CustomTemplate): string {
  return template.name;
}

export function getTemplateDescription(
  template: Template | CustomTemplate
): string | undefined {
  return template.description;
}

export function isPublicTemplate(template: Template | CustomTemplate): boolean {
  if (isCustomTemplate(template)) {
    return template.is_public;
  }
  return true; // Default templates are always public
}

export function getTemplateConfig(template: Template | CustomTemplate) {
  return template.config;
}

