import type { Category } from '../types/api';

interface ManualGenerationOptions {
  problemDescription?: string;
  category?: Category;
  model?: string;
  includeSafety?: boolean;
  includeTools?: boolean;
  includeMaterials?: boolean;
}

export const buildManualFormData = (
  options: ManualGenerationOptions,
  files?: File | File[]
): FormData => {
  const formData = new FormData();

  if (options.problemDescription) {
    formData.append('problem_description', options.problemDescription);
  }

  if (files) {
    if (Array.isArray(files)) {
      files.forEach((file) => {
        formData.append('files', file);
      });
    } else {
      formData.append('file', files);
    }
  }

  if (options.category) {
    formData.append('category', options.category);
  }

  if (options.model) {
    formData.append('model', options.model);
  }

  formData.append('include_safety', String(options.includeSafety ?? true));
  formData.append('include_tools', String(options.includeTools ?? true));
  formData.append('include_materials', String(options.includeMaterials ?? true));

  return formData;
};

