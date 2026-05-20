/**
 * Memes API
 * Handles all meme-related API operations
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { Meme, MemeCreate } from '@/types';

export interface MemeFilters {
  category?: string;
  tags?: string;
  query?: string;
}

/**
 * Get all memes with optional filters
 * @param filters - Optional filters (category, tags, query)
 * @returns Array of memes
 */
export const getAllMemes = async (filters?: MemeFilters): Promise<Meme[]> => {
  return apiGet<Meme[]>(API_ENDPOINTS.MEMES, {
    params: filters,
  });
};

/**
 * Get a single meme by ID
 * @param memeId - The meme ID
 * @returns Meme data
 */
export const getMemeById = async (memeId: string): Promise<Meme> => {
  return apiGet<Meme>(`${API_ENDPOINTS.MEMES}/${memeId}`);
};

/**
 * Create a new meme
 * @param formData - FormData containing the meme image and metadata
 * @returns Created meme
 */
export const createMeme = async (formData: FormData): Promise<Meme> => {
  return apiPost<Meme>(API_ENDPOINTS.MEMES, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

/**
 * Update an existing meme
 * @param memeId - The meme ID
 * @param meme - Meme update data
 * @returns Updated meme
 */
export const updateMeme = async (memeId: string, meme: MemeCreate): Promise<Meme> => {
  return apiPut<Meme>(`${API_ENDPOINTS.MEMES}/${memeId}`, meme);
};

/**
 * Delete a meme
 * @param memeId - The meme ID
 * @returns Deletion result
 */
export const deleteMeme = async (memeId: string): Promise<void> => {
  return apiDelete<void>(`${API_ENDPOINTS.MEMES}/${memeId}`);
};

/**
 * Get a random meme
 * @param category - Optional category filter
 * @returns Random meme
 */
export const getRandomMeme = async (category?: string): Promise<Meme> => {
  return apiGet<Meme>(`${API_ENDPOINTS.MEMES}/random`, {
    params: category ? { category } : undefined,
  });
};


