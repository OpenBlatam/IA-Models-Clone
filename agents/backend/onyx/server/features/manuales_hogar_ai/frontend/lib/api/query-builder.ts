import type { Category } from '../types/api';

interface QueryParams {
  category?: Category;
  search?: string;
  limit?: number;
  offset?: number;
  days?: number;
  interval?: string;
  threshold?: number;
  minRatings?: number;
  userId?: string;
  [key: string]: string | number | undefined;
}

export const buildQueryString = (params: QueryParams): string => {
  const urlParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      urlParams.append(key, String(value));
    }
  });

  return urlParams.toString();
};

