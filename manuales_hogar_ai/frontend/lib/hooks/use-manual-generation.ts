import { useCallback } from 'react';
import { useGenerateFromText, useGenerateFromImage, useGenerateCombined, useGenerateFromMultipleImages } from './use-manuals';
import { handleFormSuccess, handleFormError } from '../utils/form-helpers';
import type { Category } from '../types/api';

interface ManualGenerationOptions {
  problemDescription?: string;
  category?: Category;
  model?: string;
  includeSafety?: boolean;
  includeTools?: boolean;
  includeMaterials?: boolean;
}

interface UseManualGenerationReturn {
  generateFromText: (options: ManualGenerationOptions & { problemDescription: string }) => Promise<void>;
  generateFromImage: (file: File, options?: ManualGenerationOptions) => Promise<void>;
  generateFromMultipleImages: (files: File[], options?: ManualGenerationOptions) => Promise<void>;
  generateCombined: (problemDescription: string, file?: File, options?: ManualGenerationOptions) => Promise<void>;
  isLoading: boolean;
}

export const useManualGeneration = (
  onSuccess?: () => void
): UseManualGenerationReturn => {
  const generateFromTextMutation = useGenerateFromText();
  const generateFromImageMutation = useGenerateFromImage();
  const generateCombinedMutation = useGenerateCombined();
  const generateFromMultipleImagesMutation = useGenerateFromMultipleImages();

  const generateFromText = useCallback(async (options: ManualGenerationOptions & { problemDescription: string }): Promise<void> => {
    try {
      const result = await generateFromTextMutation.mutateAsync({
        problem_description: options.problemDescription,
        category: options.category,
        model: options.model,
        include_safety: options.includeSafety ?? true,
        include_tools: options.includeTools ?? true,
        include_materials: options.includeMaterials ?? true,
      });

      if (result.success && result.manual) {
        handleFormSuccess({
          successMessage: 'Manual generado exitosamente',
          onSuccess,
        });
      } else {
        handleFormError({
          error: new Error(result.error || 'Error al generar el manual'),
        });
      }
    } catch (error) {
      handleFormError({ error });
    }
  }, [generateFromTextMutation, onSuccess]);

  const generateFromImage = useCallback(async (file: File, options?: ManualGenerationOptions): Promise<void> => {
    try {
      const result = await generateFromImageMutation.mutateAsync({
        file,
        problemDescription: options?.problemDescription,
        category: options?.category,
        model: options?.model,
        includeSafety: options?.includeSafety ?? true,
        includeTools: options?.includeTools ?? true,
        includeMaterials: options?.includeMaterials ?? true,
      });

      if (result.success && result.manual) {
        handleFormSuccess({
          successMessage: 'Manual generado exitosamente',
          onSuccess,
        });
      } else {
        handleFormError({
          error: new Error(result.error || 'Error al generar el manual'),
        });
      }
    } catch (error) {
      handleFormError({ error });
    }
  }, [generateFromImageMutation, onSuccess]);

  const generateFromMultipleImages = useCallback(async (files: File[], options?: ManualGenerationOptions): Promise<void> => {
    try {
      const result = await generateFromMultipleImagesMutation.mutateAsync({
        files,
        problemDescription: options?.problemDescription,
        category: options?.category,
        model: options?.model,
        includeSafety: options?.includeSafety ?? true,
        includeTools: options?.includeTools ?? true,
        includeMaterials: options?.includeMaterials ?? true,
      });

      if (result.success && result.manual) {
        handleFormSuccess({
          successMessage: 'Manual generado exitosamente',
          onSuccess,
        });
      } else {
        handleFormError({
          error: new Error(result.error || 'Error al generar el manual'),
        });
      }
    } catch (error) {
      handleFormError({ error });
    }
  }, [generateFromMultipleImagesMutation, onSuccess]);

  const generateCombined = useCallback(async (
    problemDescription: string,
    file?: File,
    options?: ManualGenerationOptions
  ): Promise<void> => {
    try {
      const result = await generateCombinedMutation.mutateAsync({
        problemDescription,
        file,
        category: options?.category || 'general',
        model: options?.model,
        includeSafety: options?.includeSafety ?? true,
        includeTools: options?.includeTools ?? true,
        includeMaterials: options?.includeMaterials ?? true,
      });

      if (result.success && result.manual) {
        handleFormSuccess({
          successMessage: 'Manual generado exitosamente',
          onSuccess,
        });
      } else {
        handleFormError({
          error: new Error(result.error || 'Error al generar el manual'),
        });
      }
    } catch (error) {
      handleFormError({ error });
    }
  }, [generateCombinedMutation, onSuccess]);

  const isLoading =
    generateFromTextMutation.isPending ||
    generateFromImageMutation.isPending ||
    generateCombinedMutation.isPending ||
    generateFromMultipleImagesMutation.isPending;

  return {
    generateFromText,
    generateFromImage,
    generateFromMultipleImages,
    generateCombined,
    isLoading,
  };
};

