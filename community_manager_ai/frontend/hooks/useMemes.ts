/**
 * Memes Hooks
 * React Query hooks for meme-related operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { memesApi } from '@/lib/api';
import { toast } from 'sonner';
import { QUERY_KEYS } from '@/lib/config/constants';
import { getErrorMessage } from '@/lib/errors/handler';

export interface MemeFilters {
  category?: string;
  tags?: string;
  query?: string;
}

/**
 * Hook to fetch all memes with optional filters
 * @param filters - Optional filters (category, tags, query)
 * @returns React Query result with memes data
 */
export const useMemes = (filters?: MemeFilters) => {
  return useQuery({
    queryKey: QUERY_KEYS.memes.list(filters || {}),
    queryFn: () => memesApi.getAllMemes(filters),
  });
};

/**
 * Hook to fetch a single meme by ID
 * @param memeId - The meme ID
 * @returns React Query result with meme data
 */
export const useMeme = (memeId: string) => {
  return useQuery({
    queryKey: QUERY_KEYS.memes.detail(memeId),
    queryFn: () => memesApi.getMemeById(memeId),
    enabled: !!memeId,
  });
};

/**
 * Hook to create a new meme
 * @returns Mutation hook for creating memes
 */
export const useCreateMeme = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (formData: FormData) => memesApi.createMeme(formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.memes.all });
      toast.success('Meme subido exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al subir el meme');
    },
  });
};

/**
 * Hook to update an existing meme
 * @returns Mutation hook for updating memes
 */
export const useUpdateMeme = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ memeId, meme }: { memeId: string; meme: { caption?: string; tags?: string[]; category?: string } }) =>
      memesApi.updateMeme(memeId, meme),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.memes.all });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.memes.detail(variables.memeId) });
      toast.success('Meme actualizado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al actualizar el meme');
    },
  });
};

/**
 * Hook to delete a meme
 * @returns Mutation hook for deleting memes
 */
export const useDeleteMeme = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (memeId: string) => memesApi.deleteMeme(memeId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.memes.all });
      toast.success('Meme eliminado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al eliminar el meme');
    },
  });
};

/**
 * Hook to fetch a random meme
 * @param category - Optional category filter
 * @returns React Query result with random meme data
 */
export const useRandomMeme = (category?: string) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.memes.all, 'random', category],
    queryFn: () => memesApi.getRandomMeme(category),
    enabled: false, // Only executes manually via refetch
  });
};


