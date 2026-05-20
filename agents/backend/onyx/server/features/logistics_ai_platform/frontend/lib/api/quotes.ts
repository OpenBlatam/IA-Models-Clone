import { apiClient } from '../api-client';
import type { QuoteRequest, QuoteResponse } from '@/types/api';

export const quotesApi = {
  create: async (data: QuoteRequest): Promise<QuoteResponse> => {
    const response = await apiClient.post<QuoteResponse>('/forwarding/quotes', data);
    return response.data;
  },

  getAll: async (): Promise<QuoteResponse[]> => {
    try {
      const response = await apiClient.get<QuoteResponse[]>('/forwarding/quotes');
      return response.data;
    } catch (error) {
      console.error('Error fetching quotes:', error);
      return [];
    }
  },

  getById: async (quoteId: string): Promise<QuoteResponse> => {
    const response = await apiClient.get<QuoteResponse>(`/forwarding/quotes/${quoteId}`);
    return response.data;
  },
};

