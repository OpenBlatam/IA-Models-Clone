/**
 * Posts API
 * Handles all post-related API operations
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { Post, PostCreate } from '@/types';

/**
 * Get all posts with optional status filter
 * @param status - Optional status filter
 * @returns Array of posts
 */
export const getAllPosts = async (status?: string): Promise<Post[]> => {
  return apiGet<Post[]>(API_ENDPOINTS.POSTS, {
    params: status ? { status } : undefined,
  });
};

/**
 * Get a single post by ID
 * @param postId - The post ID
 * @returns Post data
 */
export const getPostById = async (postId: string): Promise<Post> => {
  return apiGet<Post>(`${API_ENDPOINTS.POSTS}/${postId}`);
};

/**
 * Create a new post
 * @param post - Post creation data
 * @returns Created post
 */
export const createPost = async (post: PostCreate): Promise<Post> => {
  return apiPost<Post>(API_ENDPOINTS.POSTS, post);
};

/**
 * Update an existing post
 * @param postId - The post ID
 * @param post - Post update data
 * @returns Updated post
 */
export const updatePost = async (postId: string, post: PostCreate): Promise<Post> => {
  return apiPut<Post>(`${API_ENDPOINTS.POSTS}/${postId}`, post);
};

/**
 * Delete a post
 * @param postId - The post ID
 * @returns Deletion result
 */
export const deletePost = async (postId: string): Promise<void> => {
  return apiDelete<void>(`${API_ENDPOINTS.POSTS}/${postId}`);
};

/**
 * Publish a post
 * @param postId - The post ID
 * @returns Published post
 */
export const publishPost = async (postId: string): Promise<Post> => {
  return apiPost<Post>(`${API_ENDPOINTS.POSTS}/${postId}/publish`);
};

/**
 * Cancel a scheduled post
 * @param postId - The post ID
 * @returns Cancelled post
 */
export const cancelPost = async (postId: string): Promise<Post> => {
  return apiPost<Post>(`${API_ENDPOINTS.POSTS}/${postId}/cancel`);
};


