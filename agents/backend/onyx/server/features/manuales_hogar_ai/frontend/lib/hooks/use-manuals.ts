import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { useQueryInvalidation } from './use-query-invalidation';
import type {
  ManualTextRequest,
  ManualListItem,
  ManualDetailResponse,
  Category,
  RatingRequest,
} from '../types/api';

export const useHealth = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30000,
  });
};

export const useModels = () => {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => apiClient.getModels(),
  });
};

export const useCategories = () => {
  return useQuery({
    queryKey: ['categories'],
    queryFn: () => apiClient.getCategories(),
  });
};

export const useGenerateFromText = () => {
  const { invalidateManuals } = useQueryInvalidation();
  return useMutation({
    mutationFn: (request: ManualTextRequest) =>
      apiClient.generateFromText(request),
    onSuccess: invalidateManuals,
  });
};

export const useGenerateFromImage = () => {
  const { invalidateManuals } = useQueryInvalidation();
  return useMutation({
    mutationFn: ({
      file,
      problemDescription,
      category,
      model,
      includeSafety,
      includeTools,
      includeMaterials,
    }: {
      file: File;
      problemDescription?: string;
      category?: Category;
      model?: string;
      includeSafety?: boolean;
      includeTools?: boolean;
      includeMaterials?: boolean;
    }) =>
      apiClient.generateFromImage(
        file,
        problemDescription,
        category,
        model,
        includeSafety,
        includeTools,
        includeMaterials
      ),
    onSuccess: invalidateManuals,
  });
};

export const useGenerateCombined = () => {
  const { invalidateManuals } = useQueryInvalidation();
  return useMutation({
    mutationFn: ({
      problemDescription,
      file,
      category,
      model,
      includeSafety,
      includeTools,
      includeMaterials,
    }: {
      problemDescription: string;
      file?: File;
      category?: Category;
      model?: string;
      includeSafety?: boolean;
      includeTools?: boolean;
      includeMaterials?: boolean;
    }) =>
      apiClient.generateCombined(
        problemDescription,
        file,
        category || 'general',
        model,
        includeSafety,
        includeTools,
        includeMaterials
      ),
    onSuccess: invalidateManuals,
  });
};

export const useGenerateFromMultipleImages = () => {
  const { invalidateManuals } = useQueryInvalidation();
  return useMutation({
    mutationFn: ({
      files,
      problemDescription,
      category,
      model,
      includeSafety,
      includeTools,
      includeMaterials,
    }: {
      files: File[];
      problemDescription?: string;
      category?: Category;
      model?: string;
      includeSafety?: boolean;
      includeTools?: boolean;
      includeMaterials?: boolean;
    }) =>
      apiClient.generateFromMultipleImages(
        files,
        problemDescription,
        category,
        model,
        includeSafety,
        includeTools,
        includeMaterials
      ),
    onSuccess: invalidateManuals,
  });
};

export const useManuals = (
  category?: Category,
  search?: string,
  limit = 20,
  offset = 0
) => {
  return useQuery({
    queryKey: ['manuals', category, search, limit, offset],
    queryFn: () => apiClient.getManuals(category, search, limit, offset),
  });
};

export const useManual = (id: number) => {
  return useQuery({
    queryKey: ['manual', id],
    queryFn: () => apiClient.getManual(id),
    enabled: !!id,
  });
};

export const useRecentManuals = (limit = 10, category?: Category) => {
  return useQuery({
    queryKey: ['recent-manuals', limit, category],
    queryFn: () => apiClient.getRecentManuals(limit, category),
  });
};

export const useStatistics = (days = 30) => {
  return useQuery({
    queryKey: ['statistics', days],
    queryFn: () => apiClient.getStatistics(days),
  });
};

export const useSearch = (
  query: string,
  category?: Category,
  limit = 20,
  offset = 0
) => {
  return useQuery({
    queryKey: ['search', query, category, limit, offset],
    queryFn: () => apiClient.search(query, category, limit, offset),
    enabled: query.length > 0,
  });
};

export const useAddRating = () => {
  const { invalidateManual } = useQueryInvalidation();
  return useMutation({
    mutationFn: ({
      manualId,
      request,
      userId,
    }: {
      manualId: number;
      request: RatingRequest;
      userId?: string;
    }) => apiClient.addRating(manualId, request, userId),
    onSuccess: (_, variables) => {
      invalidateManual(variables.manualId);
    },
  });
};

export const useRatings = (manualId: number, limit = 20, offset = 0) => {
  return useQuery({
    queryKey: ['ratings', manualId, limit, offset],
    queryFn: () => apiClient.getRatings(manualId, limit, offset),
    enabled: !!manualId,
  });
};

export const useAddFavorite = () => {
  const { invalidateFavorites, invalidateManual } = useQueryInvalidation();
  return useMutation({
    mutationFn: ({ manualId, userId }: { manualId: number; userId: string }) =>
      apiClient.addFavorite(manualId, userId),
    onSuccess: (_, variables) => {
      invalidateFavorites(variables.userId);
      invalidateManual(variables.manualId);
    },
  });
};

export const useRemoveFavorite = () => {
  const { invalidateFavorites, invalidateManual } = useQueryInvalidation();
  return useMutation({
    mutationFn: ({ manualId, userId }: { manualId: number; userId: string }) =>
      apiClient.removeFavorite(manualId, userId),
    onSuccess: (_, variables) => {
      invalidateFavorites(variables.userId);
      invalidateManual(variables.manualId);
    },
  });
};

export const useCheckFavorite = (manualId: number, userId: string) => {
  return useQuery({
    queryKey: ['favorite', manualId, userId],
    queryFn: () => apiClient.checkFavorite(manualId, userId),
    enabled: !!manualId && !!userId,
  });
};

