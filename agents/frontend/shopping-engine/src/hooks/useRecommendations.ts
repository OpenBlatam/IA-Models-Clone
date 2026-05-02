'use client';

import apiClient from '@/src/api/client';
import { useApiTask } from './useApiTask';
import type { GetRecommendationsRequest, RecommendationsResult } from '@/src/types/api';

export interface UseRecommendationsReturn {
    getRecommendations: (request: GetRecommendationsRequest) => Promise<void>;
    result: RecommendationsResult | null;
    isLoading: boolean;
    isPolling: boolean;
    error: string | null;
    reset: () => void;
}

export const useRecommendations = (): UseRecommendationsReturn => {
    const { execute, result, isLoading, isPolling, error, reset } = useApiTask<GetRecommendationsRequest, RecommendationsResult>(
        (req) => apiClient.getRecommendations(req),
        'Getting recommendations failed'
    );

    return {
        getRecommendations: execute,
        result,
        isLoading,
        isPolling,
        error,
        reset,
    };
};

export default useRecommendations;
