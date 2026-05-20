/**
 * Template module specific types
 */

import type { Template, CustomTemplate } from '@/types/api';

export interface TemplateCardProps {
  template: Template | CustomTemplate;
  onPress?: () => void;
  selected?: boolean;
}

export interface TemplateListProps {
  templates: (Template | CustomTemplate)[];
  isLoading?: boolean;
  onTemplatePress?: (template: Template | CustomTemplate) => void;
  selectedTemplate?: string;
}

export interface TemplateSelectorProps {
  onSelect: (template: Template | CustomTemplate) => void;
  selectedTemplate?: string;
}

