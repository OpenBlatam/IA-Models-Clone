/**
 * Posts Hooks
 * React Query hooks for post-related operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { postsApi } from '@/lib/api';
import { PostCreate } from '@/types';
import { toast } from 'sonner';
import { QUERY_KEYS } from '@/lib/config/constants';
import { getErrorMessage } from '@/lib/errors/handler';

/**
 * Hook to fetch all posts with optional status filter
 * @param status - Optional status filter
 * @returns React Query result with posts data
 */
export const usePosts = (status?: string) => {
  return useQuery({
    queryKey: QUERY_KEYS.posts.list({ status }),
    queryFn: () => postsApi.getAllPosts(status),
  });
};

/**
 * Hook to fetch a single post by ID
 * @param postId - The post ID
 * @returns React Query result with post data
 */
export const usePost = (postId: string) => {
  return useQuery({
    queryKey: QUERY_KEYS.posts.detail(postId),
    queryFn: () => postsApi.getPostById(postId),
    enabled: !!postId,
  });
};

/**
 * Hook to create a new post
 * @returns Mutation hook for creating posts
 */
export const useCreatePost = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (post: PostCreate) => postsApi.createPost(post),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.posts.all });
      toast.success('Post creado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al crear el post');
    },
  });
};

/**
 * Hook to update an existing post
 * @returns Mutation hook for updating posts
 */
export const useUpdatePost = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ postId, post }: { postId: string; post: PostCreate }) =>
      postsApi.updatePost(postId, post),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.posts.all });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.posts.detail(variables.postId) });
      toast.success('Post actualizado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al actualizar el post');
    },
  });
};

/**
 * Hook to delete a post
 * @returns Mutation hook for deleting posts
 */
export const useDeletePost = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (postId: string) => postsApi.deletePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.posts.all });
      toast.success('Post eliminado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al eliminar el post');
    },
  });
};

/**
 * Hook to publish a post
 * @returns Mutation hook for publishing posts
 */
export const usePublishPost = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (postId: string) => postsApi.publishPost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.posts.all });
      toast.success('Post publicado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al publicar el post');
    },
  });
};

/**
 * Hook to cancel a scheduled post
 * @returns Mutation hook for cancelling posts
 */
export const useCancelPost = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (postId: string) => postsApi.cancelPost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.posts.all });
      toast.success('Post cancelado exitosamente');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error) || 'Error al cancelar el post');
    },
  });
};


